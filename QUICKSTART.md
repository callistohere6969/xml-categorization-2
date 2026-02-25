# Quick Start Guide - XML Error Categorization (Hybrid AI)

## ⚡ Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up API Keys

Edit the [.env](.env) file with your credentials:

```bash
# Required: OpenAI for reasoning
OPENAI_API_KEY=sk-xxxxxxxxxxxx

# Required: Langfuse for usage tracking (provided by your office)
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: OpenRouter for cheaper embeddings
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
```

**Where to get API keys:**
- OpenAI: https://platform.openai.com/api-keys
- Langfuse: Your office admin provides these
- OpenRouter (optional): https://openrouter.ai/keys

### Step 3: Prepare Your Data

Edit `new_errors.csv` with your XML errors. Format:
```csv
normalized_error
"Your first XML error message here"
"Your second XML error message here"
```

### Step 4: Run Classification

```bash
python step2_classify_hybrid.py
```

### Step 5: Check Results

Open `categorized_output_hybrid.csv` to see:
- Categorized errors
- Confidence scores
- Whether reasoning model was used
- AI reasoning explanation (for complex cases)

## 🎛️ Configuration Options

### Model Selection

**Reasoning Models (for edge cases):**
```bash
export OPENAI_REASONING_MODEL=gpt-4o          # Recommended: best balance
export OPENAI_REASONING_MODEL=gpt-4-turbo     # Faster, cheaper
export OPENAI_REASONING_MODEL=o1-preview      # Most advanced, expensive
```

### Threshold Tuning

**When to use reasoning model:**
```bash
export CONFIDENCE_THRESHOLD=0.55  # Default
export CONFIDENCE_THRESHOLD=0.70  # Less reasoning (lower cost)
export CONFIDENCE_THRESHOLD=0.40  # More reasoning (higher accuracy)
```

## 📊 Understanding Output

The output CSV contains:
- `normalized_error`: Your original error
- `final_category_id`: Category ID (0-10)
- `category_name`: Human-readable category
- `confidence`: Classification confidence (0-1)
- `reasoning_used`: True if AI reasoning was used
- `reasoning_explanation`: AI's explanation (when applicable)

## 💰 Cost Estimation

**Typical costs for 10,000 errors:**
- Embeddings (OpenRouter): ~$0.01 (all errors) - or use OpenAI for $0.01
- Reasoning (OpenAI): ~$0.05-0.20 (5-15% of errors)
- **Total**: ~$0.06-0.21

**Langfuse tracks all costs automatically** - check your dashboard for real-time spending!

## 🔍 Monitoring

Watch the console output while running:
- "⚡ Low confidence..." - Reasoning model activated
- Category distribution at the end
- Reasoning usage percentage

**View detailed analytics in Langfuse:**
- Visit your Langfuse dashboard (LANGFUSE_HOST from your .env)
- See all API calls, costs, and latencies
- Debug specific classifications
- Track usage over time

## 🐛 Troubleshooting

**"OPENAI_API_KEY not found"**
→ Add to .env file: `OPENAI_API_KEY=your_key`

**"Langfuse tracking not configured"**
→ Add Langfuse credentials to .env file (provided by your office)

**Too many reasoning calls (>20%)**
→ Increase `CONFIDENCE_THRESHOLD` to 0.65 or 0.70 in .env

**Not enough reasoning calls (<5%)**
→ Decrease `CONFIDENCE_THRESHOLD` to 0.45 or 0.40 in .env

**Can't see usage in Langfuse**
→ Check that LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY are set correctly

## 📚 Next Steps

1. Test with sample data: `python step2_classify_hybrid.py`
2. Review results in `categorized_output_hybrid.csv`
3. Adjust thresholds if needed
4. Process your full dataset
5. Monitor costs in your API dashboards

---

**Need Help?** Check the main [README.md](README.md) for detailed documentation.
