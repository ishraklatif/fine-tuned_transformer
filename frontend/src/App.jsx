import { useState } from "react";

const CLASS_DESCRIPTIONS = {
  ABBR: "Abbreviation — questions about what something stands for",
  DESC: "Description — questions asking for definitions or explanations",
  ENTY: "Entity — questions about things, substances, or objects",
  HUM: "Human — questions about people or organisations",
  LOC: "Location — questions about places",
  NUM: "Numeric — questions about quantities, measurements, or dates",
};

const CLASS_COLORS = {
  ABBR: "bg-purple-100 text-purple-800 border-purple-300",
  DESC: "bg-blue-100 text-blue-800 border-blue-300",
  ENTY: "bg-green-100 text-green-800 border-green-300",
  HUM: "bg-orange-100 text-orange-800 border-orange-300",
  LOC: "bg-red-100 text-red-800 border-red-300",
  NUM: "bg-yellow-100 text-yellow-800 border-yellow-300",
};

const API_URL = import.meta?.env?.VITE_API_URL ?? "http://localhost:8000";

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const examples = [
    "What does NASA stand for?",
    "Who invented the telephone?",
    "Where is the Eiffel Tower located?",
    "How many planets are in the solar system?",
    "What is photosynthesis?",
    "What kind of animal is a dolphin?",
  ];

  async function classify(q) {
    const text = q ?? question;
    if (!text.trim()) return;
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

  const maxScore = result
    ? Math.max(...Object.values(result.all_scores))
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 text-white flex flex-col items-center px-4 py-12">
      {/* Header */}
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold tracking-tight mb-2">
          🧠 Question Classifier
        </h1>
        <p className="text-slate-400 text-sm max-w-md">
          Powered by BERT prefix tuning — classifies questions into 6 semantic categories
        </p>
      </div>

      {/* Input */}
      <div className="w-full max-w-2xl bg-slate-800 rounded-2xl p-6 shadow-xl mb-6">
        <label className="block text-sm text-slate-400 mb-2">Enter a question</label>
        <textarea
          className="w-full bg-slate-700 rounded-xl p-4 text-white text-base resize-none outline-none focus:ring-2 focus:ring-blue-500 placeholder-slate-500"
          rows={3}
          placeholder="e.g. What does NASA stand for?"
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
          onClick={() => classify()}
          disabled={loading || !question.trim()}
          className="mt-4 w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed font-semibold text-base transition-all"
        >
          {loading ? "Classifying…" : "Classify Question"}
        </button>
      </div>

      {/* Examples */}
      <div className="w-full max-w-2xl mb-6">
        <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider">Try an example</p>
        <div className="flex flex-wrap gap-2">
          {examples.map((ex) => (
            <button
              key={ex}
              onClick={() => { setQuestion(ex); classify(ex); }}
              className="text-xs bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded-full transition-all text-slate-300"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="w-full max-w-2xl bg-red-900/50 border border-red-700 text-red-300 rounded-xl p-4 mb-6 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="w-full max-w-2xl bg-slate-800 rounded-2xl p-6 shadow-xl">
          {/* Predicted class badge */}
          <div className="flex items-center gap-3 mb-6">
            <span
              className={`border px-4 py-1.5 rounded-full text-sm font-bold ${CLASS_COLORS[result.predicted_class] ?? "bg-slate-200 text-slate-800"}`}
            >
              {result.predicted_class}
            </span>
            <span className="text-slate-400 text-sm">
              {(result.confidence * 100).toFixed(1)}% confidence
            </span>
          </div>

          <p className="text-slate-400 text-sm mb-5">
            {CLASS_DESCRIPTIONS[result.predicted_class]}
          </p>

          {/* Score bars */}
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">All scores</p>
          <div className="space-y-2">
            {Object.entries(result.all_scores)
              .sort(([, a], [, b]) => b - a)
              .map(([cls, score]) => (
                <div key={cls} className="flex items-center gap-3">
                  <span className="w-12 text-xs font-mono text-slate-400 text-right">{cls}</span>
                  <div className="flex-1 bg-slate-700 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-2 rounded-full transition-all ${score === maxScore ? "bg-blue-500" : "bg-slate-500"}`}
                      style={{ width: `${(score * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className="w-12 text-xs font-mono text-slate-400">
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      <p className="mt-10 text-slate-600 text-xs">
        FIT5215 Deep Learning · Assignment 2 · Monash University
      </p>
    </div>
  );
}
