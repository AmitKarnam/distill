//go:build postgres

package memory

import (
	"database/sql"
	"fmt"

	"github.com/Siddhant-K-code/distill/pkg/sensitivity"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// PostgresStore uses a connection pool (pgxpool) and relies on Postgres's MVCC for concurrency safety. Deduplication uses advisory locks to prevent TOCTOU races.

type PostgresStore struct {
	dbPool *pgxpool.Pool
	cfg        Config
	handlers   []MemoryEventHandler
	classifier *sensitivity.Classifier
}

func NewPostgresStore(dsn string, cfg Config) (*PostgresStore, error) {
	if dsn == "" {
		return nil, fmt.Errorf("empty DSN")
	}

    poolCfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("postgres: parse dsn: %w", err)
	}
 
	pool, err := pgxpool.NewWithConfig(ctx, poolCfg)
	if err != nil {
		return nil, fmt.Errorf("postgres: create pool: %w", err)
	}
 
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("postgres: ping: %w", err)
	}
 
	s := &PostgresStore{
		pool:       pool,
		cfg:        cfg,
		classifier: sensitivity.New(sensitivity.DefaultConfig()),
		decayDone:  make(chan struct{}),
	}
 
	if err := s.migrate(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("postgres: migrate: %w", err)
	}

	return s, nil
}

func (ps *PostgresStore) migrate() error {
	schema := `
	CREATE TABLE IF NOT EXISTS memories (
  		id TEXT PRIMARY KEY,
  		text TEXT NOT NULL,
  		embedding BYTEA,
  		source TEXT DEFAULT '',
  		session_id TEXT DEFAULT '',
 		metadata TEXT DEFAULT '{}',
  		decay_level INTEGER DEFAULT 0,
  		sensitivity INTEGER DEFAULT 0,
  		created_at TIMESTAMPTZ NOT NULL,
  		last_referenced TIMESTAMPTZ NOT NULL,
  		access_count INTEGER DEFAULT 0,
  		expired BOOLEAN DEFAULT FALSE,
  		expired_at TIMESTAMPTZ,
  		superseded_by TEXT DEFAULT '',
  		expires_at TIMESTAMPTZ
	);
	CREATE TABLE IF NOT EXISTS memory_tags (
  		memory_id TEXT NOT NULL,
  		tag TEXT NOT NULL,
  		PRIMARY KEY (memory_id, tag)
	);
	CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag);
    CREATE INDEX IF NOT EXISTS idx_memories_decay ON memories(decay_level);
    CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
    CREATE INDEX IF NOT EXISTS idx_memories_referenced ON memories(last_referenced);
    CREATE INDEX IF NOT EXISTS idx_memories_expired ON memories(expired);
	`
	_, err := s.pool.Exec(ctx, schema)
	return err
}

func (ps *PostgresStore) Store(ctx context.Context, req StoreRequest) (*StoreResult, error) {
	result := &StoreResult{}

	for _,entry := req.Entries {
		if entry.Text == "" {
			continue
		}

		if len(entry.Embedding) > 0 {
			similar,err := ps.findSimilar(ctx,entry.Embedding)
			if err != nil {
				return nil, fmt.Errorf("find similar: %w", err)
			}

			isDup := false
			for _,sim := range similar{
				if sim.isDup {
					_, err := s.pool.Exec(ctx,
						`UPDATE memories SET last_referenced = NOW(), access_count = access_count + 1 WHERE id = $1`,
						sim.id,
					)
					if err != nil {
						return nil, fmt.Errorf("postgres: update duplicate: %w", err)
					}
					result.Deduplicated++
					isDup = true
					break
				}
			}

			if isDup {
				continue
			}

			// handle conflicts
			for _, sim := range similar {
				result.Conflicts = append(result.Conflicts, Conflict{
					NewText:      entry.Text,
					ExistingID:   sim.id,
					ExistingText: sim.text,
					Distance:     sim.distance,
				})
			}
		}

		id := generateID()
		metaJSON, _ := json.Marshal(entry.Metadata)
		embBlob := encodeEmbedding(entry.Embedding)
 
		sens := entry.Sensitivity
		if entry.AutoClassify {
			classified := s.classifier.Classify(entry.Text)
			if classified.Level > sens {
				sens = classified.Level
			}
		}
 
		expiresAt := ""
		if entry.ExpiresAt != nil {
			expiresAt = entry.ExpiresAt.UTC().Format(time.RFC3339Nano)
		}

		_, err := s.pool.Exec(ctx, `
			INSERT INTO memories
				(id, text, embedding, source, session_id, metadata, decay_level, sensitivity,
				 created_at, last_referenced, access_count, expires_at)
			VALUES ($1,$2,$3,$4,$5,$6,0,$7,NOW(),NOW(),0,$8)`,
			id, entry.Text, embBlob, entry.Source, req.SessionID,
			string(metaJSON), int(sens), expiresAt,
		)
		if err != nil {
			return nil, fmt.Errorf("postgres: insert memory: %w", err)
		}
		
		for _, tag := range entry.Tags {
			_, err := s.pool.Exec(ctx,
				`INSERT INTO memory_tags (memory_id, tag) VALUES ($1,$2) ON CONFLICT DO NOTHING`,
				id, tag,
			)
			if err != nil {
				return nil, fmt.Errorf("postgres: insert tag: %w", err)
			}
		}
 
		for i := range result.Conflicts {
			if result.Conflicts[i].NewID == "" {
				result.Conflicts[i].NewID = id
			}
		}
 
		result.Stored++
	}

	var total int
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM memories`).Scan(&total); err != nil {
		return nil, err
	}
	result.TotalMemories = total
 
	return result, nil
}

type pgSimilarEntry struct {
	id       string
	text     string
	distance float64
	isDup    bool
}
 
// findSimilar performs a full-scan cosine distance search.
// The comment in the SQLite implementation applies equally here
// for < 10K rows; at larger scale consider pgvector or a separate ANN index.
func (s *PostgresStore) findSimilar(ctx context.Context, embedding []float32) ([]pgSimilarEntry, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT id, text, embedding FROM memories WHERE embedding IS NOT NULL AND expired = FALSE`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
 
	conflictThreshold := s.cfg.ConflictThreshold
	if conflictThreshold <= 0 {
		conflictThreshold = 0.35
	}
 
	var results []pgSimilarEntry
	for rows.Next() {
		var id, text string
		var embBlob []byte
		if err := rows.Scan(&id, &text, &embBlob); err != nil {
			return nil, err
		}
 
		existing := decodeEmbedding(embBlob)
		if len(existing) == 0 {
			continue
		}
 
		dist := distillmath.CosineDistance(embedding, existing)
		if dist < s.cfg.DedupThreshold {
			return []pgSimilarEntry{{id: id, text: text, distance: dist, isDup: true}}, nil
		}
		if dist < conflictThreshold {
			results = append(results, pgSimilarEntry{id: id, text: text, distance: dist})
		}
	}
 
	return results, rows.Err()
}