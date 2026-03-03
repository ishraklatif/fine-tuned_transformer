import { useState } from "react";

const CLASS_DESCRIPTIONS = {
  ABBR: "Questions asking what something stands for or its full form",
  DESC: "Questions asking for definitions, explanations or descriptions",
  ENTY: "Questions about things, substances, objects or concepts",
  HUM:  "Questions about people, groups or organisations",
  LOC:  "Questions about places, cities, countries or regions",
  NUM:  "Questions about quantities, measurements, dates or counts",
};

const CLASS_COLORS = {
  ABBR: "#c77dff",
  DESC: "#48cae4",
  ENTY: "#80b918",
  HUM:  "#f4a261",
  LOC:  "#e63946",
  NUM:  "#f9c74f",
};

const EXAMPLES = [
  "What does NASA stand for?",
  "Who invented the telephone?",
  "Where is the Eiffel Tower located?",
  "How many planets are in the solar system?",
  "What is photosynthesis?",
  "What kind of animal is a dolphin?",
];

const API_URL = import.meta?.env?.VITE_API_URL ?? "http://localhost:8000";

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  async function classify(q) {
    const text = (q ?? question).trim();
    if (!text) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? "Prediction failed");
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const sorted = result
    ? Object.entries(result.all_scores).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div style={styles.root}>
      <div style={styles.bgGlow} />

      {/* Header */}
      <header style={styles.header}>
        <div style={styles.logoRow}>
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
            <circle cx="14" cy="14" r="13" stroke="#d4a96a" strokeWidth="1.5" />
            <path d="M8 14 Q14 6 20 14 Q14 22 8 14Z" fill="#d4a96a" opacity="0.85" />
          </svg>
          <span style={styles.logoText}>Question Classifier</span>
        </div>
        <p style={styles.subtitle}>
          Semantic classification powered by BERT prefix tuning
        </p>
      </header>

      {/* Main */}
      <main style={styles.main}>
        <div style={styles.card}>
          <div style={styles.inputWrapper}>
            <textarea
              style={styles.textarea}
              rows={3}
              placeholder="Ask anything — e.g. What does NASA stand for?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  classify();
                }
              }}
            />
            <button
              style={{
                ...styles.button,
                opacity: loading || !question.trim() ? 0.4 : 1,
                cursor: loading || !question.trim() ? "not-allowed" : "pointer",
              }}
              onClick={() => classify()}
              disabled={loading || !question.trim()}
            >
              {loading ? (
                <span style={styles.spinner} />
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="22" y1="2" x2="11" y2="13" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                  Classify
                </>
              )}
            </button>
          </div>

          <div style={styles.examplesRow}>
            <span style={styles.examplesLabel}>Try:</span>
            <div style={styles.pills}>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  style={styles.pill}
                  onMouseEnter={e => e.currentTarget.style.background = "rgba(212,169,106,0.12)"}
                  onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.04)"}
                  onClick={() => { setQuestion(ex); classify(ex); }}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        </div>

        {error && (
          <div style={styles.errorBox}>⚠ {error}</div>
        )}

        {result && (
          <div style={styles.resultCard}>
            <div style={styles.resultHeader}>
              <span
                style={{
                  ...styles.classBadge,
                  color: CLASS_COLORS[result.predicted_class] ?? "#fff",
                  borderColor: CLASS_COLORS[result.predicted_class] ?? "#fff",
                  boxShadow: `0 0 12px ${CLASS_COLORS[result.predicted_class]}33`,
                }}
              >
                {result.predicted_class}
              </span>
              <span style={styles.confidence}>
                {(result.confidence * 100).toFixed(1)}% confidence
              </span>
            </div>

            <p style={styles.classDesc}>
              {CLASS_DESCRIPTIONS[result.predicted_class]}
            </p>

            <div style={styles.divider} />

            <p style={styles.scoresLabel}>All scores</p>
            <div style={styles.bars}>
              {sorted.map(([cls, score]) => (
                <div key={cls} style={styles.barRow}>
                  <span style={styles.barLabel}>{cls}</span>
                  <div style={styles.barTrack}>
                    <div
                      style={{
                        ...styles.barFill,
                        width: `${(score * 100).toFixed(1)}%`,
                        background: cls === result.predicted_class
                          ? CLASS_COLORS[cls]
                          : "rgba(255,255,255,0.15)",
                      }}
                    />
                  </div>
                  <span style={styles.barValue}>
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <footer style={styles.footer}>
        FIT5215 Deep Learning · Monash University · BERT Prefix Tuning
      </footer>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        textarea:focus { border-color: rgba(212,169,106,0.4) !important; box-shadow: 0 0 0 3px rgba(212,169,106,0.08); }
        textarea::placeholder { color: #4a4038; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #1a1612; }
      `}</style>
    </div>
  );
}

const styles = {
  root: {
    minHeight: "100vh",
    background: "#1a1612",
    color: "#e8e0d4",
    fontFamily: "'Georgia', 'Times New Roman', serif",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "0 16px 48px",
    position: "relative",
    overflowX: "hidden",
  },
  bgGlow: {
    position: "fixed",
    top: "-20%",
    left: "50%",
    transform: "translateX(-50%)",
    width: "600px",
    height: "400px",
    background: "radial-gradient(ellipse, rgba(212,169,106,0.06) 0%, transparent 70%)",
    pointerEvents: "none",
    zIndex: 0,
  },
  header: {
    textAlign: "center",
    padding: "52px 0 32px",
    position: "relative",
    zIndex: 1,
  },
  logoRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "10px",
    marginBottom: "12px",
  },
  logoText: {
    fontSize: "22px",
    fontWeight: "600",
    color: "#e8e0d4",
    letterSpacing: "-0.3px",
  },
  subtitle: {
    fontSize: "14px",
    color: "#7a6f62",
    margin: 0,
    fontStyle: "italic",
  },
  main: {
    width: "100%",
    maxWidth: "660px",
    position: "relative",
    zIndex: 1,
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  card: {
    background: "#231f1a",
    border: "1px solid rgba(212,169,106,0.15)",
    borderRadius: "16px",
    padding: "20px",
  },
  inputWrapper: {
    display: "flex",
    gap: "10px",
    alignItems: "flex-end",
  },
  textarea: {
    flex: 1,
    background: "#2c2620",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "10px",
    padding: "12px 14px",
    color: "#e8e0d4",
    fontSize: "15px",
    fontFamily: "'Georgia', serif",
    resize: "none",
    outline: "none",
    lineHeight: "1.5",
    transition: "border-color 0.2s, box-shadow 0.2s",
  },
  button: {
    background: "#d4a96a",
    color: "#1a1612",
    border: "none",
    borderRadius: "10px",
    padding: "0 18px",
    height: "44px",
    fontWeight: "700",
    fontSize: "14px",
    fontFamily: "Georgia, serif",
    display: "flex",
    alignItems: "center",
    gap: "6px",
    transition: "opacity 0.2s",
    whiteSpace: "nowrap",
    flexShrink: 0,
  },
  spinner: {
    width: "14px",
    height: "14px",
    border: "2px solid rgba(26,22,18,0.3)",
    borderTop: "2px solid #1a1612",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
    display: "inline-block",
  },
  examplesRow: {
    marginTop: "14px",
    display: "flex",
    alignItems: "flex-start",
    gap: "8px",
  },
  examplesLabel: {
    fontSize: "11px",
    color: "#5a5048",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    paddingTop: "6px",
    flexShrink: 0,
  },
  pills: {
    display: "flex",
    flexWrap: "wrap",
    gap: "6px",
  },
  pill: {
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "20px",
    padding: "4px 12px",
    fontSize: "12px",
    color: "#9a8f82",
    cursor: "pointer",
    fontFamily: "Georgia, serif",
    transition: "background 0.15s",
  },
  errorBox: {
    background: "rgba(230,57,70,0.1)",
    border: "1px solid rgba(230,57,70,0.3)",
    borderRadius: "10px",
    padding: "12px 16px",
    fontSize: "13px",
    color: "#e63946",
  },
  resultCard: {
    background: "#231f1a",
    border: "1px solid rgba(212,169,106,0.15)",
    borderRadius: "16px",
    padding: "24px",
  },
  resultHeader: {
    display: "flex",
    alignItems: "center",
    gap: "14px",
    marginBottom: "10px",
  },
  classBadge: {
    border: "1px solid",
    borderRadius: "8px",
    padding: "4px 14px",
    fontSize: "13px",
    fontWeight: "700",
    letterSpacing: "0.08em",
  },
  confidence: {
    fontSize: "13px",
    color: "#7a6f62",
  },
  classDesc: {
    fontSize: "14px",
    color: "#7a6f62",
    margin: "0 0 16px",
    fontStyle: "italic",
    lineHeight: "1.5",
  },
  divider: {
    height: "1px",
    background: "rgba(255,255,255,0.06)",
    marginBottom: "16px",
  },
  scoresLabel: {
    fontSize: "11px",
    color: "#5a5048",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    marginBottom: "10px",
  },
  bars: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  barRow: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  barLabel: {
    width: "36px",
    fontSize: "11px",
    fontFamily: "monospace",
    color: "#7a6f62",
    textAlign: "right",
    flexShrink: 0,
  },
  barTrack: {
    flex: 1,
    height: "6px",
    background: "rgba(255,255,255,0.06)",
    borderRadius: "99px",
    overflow: "hidden",
  },
  barFill: {
    height: "100%",
    borderRadius: "99px",
    transition: "width 0.6s cubic-bezier(0.4,0,0.2,1)",
  },
  barValue: {
    width: "40px",
    fontSize: "11px",
    fontFamily: "monospace",
    color: "#7a6f62",
    flexShrink: 0,
  },
  footer: {
    marginTop: "48px",
    fontSize: "11px",
    color: "#3a3028",
    textAlign: "center",
    position: "relative",
    zIndex: 1,
  },
};
