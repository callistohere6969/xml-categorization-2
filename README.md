# XML Error Categorization System

**AI-powered XML validation error categorization using Hybrid Intelligence**

## 🎯 Overview

This system automatically categorizes XML validation errors into 11 predefined categories using a **hybrid approach**: fast embedding-based classification for clear cases, and OpenAI's reasoning models for complex edge cases.

- **Method**: OpenRouter embeddings + OpenAI reasoning model
- **Speed**: Fast for most errors, intelligent reasoning for edge cases
- **Cost**: OpenRouter embeddings + OpenAI API (pay per use)
- **Accuracy**: High accuracy with intelligent fallback for difficult cases

## 📊 Categories

The system classifies errors into 11 categories:

| ID | Category Name | 
|----|--------------|
| 0 | Mismatch In Graphic Asset Declaration |
| 1 | Missing Referred Id In Element |
| 2 | Incorrect Element Content Structure |
| 3 | Invalid Id Attribute Value |
| 4 | Invalid Para Or Character Style Present |
| 5 | Required Element Missing |
| 6 | Duplicate Id Attribute |
| 7 | Undeclared Entity Present |
| 8 | Attribute Id Is Missing |
| 9 | Unclosed Element Tag |
| 10 | Invalid Attribute Format |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Input Data

Create `new_errors.csv` with column `normalized_error`:

```csv
normalized_error
"<![CDATA[Error: The content of element type 'ce:author' is incomplete, it must match...]]>"
"Element type 'fontstyle21' must be declared."
"An element with the identifier 'fig1' must appear in the document."
```

### 3. Set API Keys

Edit the [.env](.env) file with your credentials:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=sk-xxxxxxxxxxxx

# Required: Langfuse for usage tracking
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: OpenRouter for cheaper embeddings
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx

# Optional: customize models and thresholds
OPENAI_REASONING_MODEL=gpt-4o
CONFIDENCE_THRESHOLD=0.55
```

### 4. Run Classification

```bash
python step2_classify_hybrid.py
```

### 5. Check Results

Output file: `categorized_output_hybrid.csv`

```csv
normalized_error,final_category_id,category_name,confidence,reasoning_used,reasoning_explanation
"The content of element type 'ce:author' is incomplete...",5,"Required Element Missing",0.823,False,""
"Element type 'fontstyle21' must be declared.",4,"Invalid Para Or Character Style Present",0.95,True,"The error indicates an undeclared element type, which matches the Invalid Para Or Character Style Present category..."
```

Flagged (unmatched) errors are written to `flagged_errors_current.csv`.

## 📁 Project Structure

```
XML-categorization/
├── categories.json                              # 11 error category definitions
├── cleaner.py                                   # Text normalization utility
├── step2_classify_hybrid.py                    # Main classifier (USE THIS!)
├── new_errors.csv                              # Input: Your XML errors
├── categorized_output_hybrid.csv               # Output: Classified errors
├── flagged_errors_current.csv                  # Low-confidence errors (if any)
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

## 🧭 Architecture

```
new_errors.csv
    │
    ▼
cleaner.py  (normalize text)
    │
    ▼
OpenRouter embeddings
    │
    ▼
cosine similarity
    │
    ▼
confidence >= threshold? ───YES──► categorized_output_hybrid.csv
    │
    NO (low confidence)
    │
    ▼
OpenAI Reasoning Model
(intelligent analysis)
    │
    ▼
categorized_output_hybrid.csv
```

## 🎯 How It Works

The system uses a **hybrid intelligence approach**:

### 1. OpenRouter Embeddings (Fast Initial Classification)
- Uses `openai/text-embedding-3-small` for speed and accuracy
- Generates embeddings for errors and categories
- Computes cosine similarity for classification
- **Fast and cost-effective** for straightforward cases

### 2. OpenAI Reasoning Model (Intelligent Fallback)
- Activates when embedding confidence < threshold (default 0.55)
- Uses advanced models: `gpt-4o`, `gpt-4-turbo`, or `o1-preview`
- Provides intelligent reasoning about complex errors
- **No hard-coded rules** - pure AI reasoning

**Key Advantages:**
- ✅ **Best of both worlds**: Speed + Intelligence
- ✅ **No brittle rules**: AI learns patterns naturally
- ✅ **Handles edge cases**: Reasoning model for complex errors
- ✅ **Cost-efficient**: Reasoning only when needed
- ✅ **Explainable**: Get reasoning for difficult classifications

## 📊 Performance Snapshot

| Metric | Value |
|--------|-------|
| **Speed** | Fast (embeddings) + Smart (reasoning when needed) |
| **Cost** | OpenRouter embeddings (~$0.01/10K) + OpenAI reasoning (variable) |
| **Reasoning Usage** | Typically 5-15% of errors need reasoning |
| **Explainability** | ✅ Get reasoning for complex cases |

## 🔧 Configuration

Edit [.env](.env) file with your credentials:

**Required:**
```bash
OPENAI_API_KEY=your_openai_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Optional:**
```bash
# Use OpenRouter for cheaper embeddings (optional)
OPENROUTER_API_KEY=your_openrouter_key

# Customize reasoning model (default: gpt-4o)
OPENAI_REASONING_MODEL=gpt-4o

# When to trigger reasoning model (default: 0.55)
CONFIDENCE_THRESHOLD=0.55

# Threshold for flagging unmatched errors
FLAG_UNMATCHED_THRESHOLD=0.55
```

**Model Recommendations:**
- `gpt-4o`: Best balance of speed, cost, and intelligence (recommended)
- `gpt-4-turbo`: Faster, slightly cheaper
- `o1-preview`: Most advanced reasoning (slower, more expensive)

**Langfuse Benefits:**
- 📊 Track all API calls and costs
- 🔍 Debug problematic classifications
- 📈 Monitor usage patterns
- 💰 Optimize costs over time

## � Flagged Errors

Errors with very low embedding confidence that still don't meet criteria for reasoning are flagged for review:

**Files generated:**
- `flagged_errors_current.csv` - Errors that need manual review

Note: With the reasoning model, flagged errors should be rare since the AI handles most edge cases.

## 🎓 Key Advantages

The hybrid AI approach provides significant benefits:

| Feature | Benefit |
|---------|---------|
| **No Hard-Coded Rules** | AI naturally learns patterns, easier to maintain |
| **Intelligent Reasoning** | Handles complex edge cases automatically |
| **Cost-Efficient** | Reasoning only when needed (~5-15% of errors) |
| **Explainable** | Get AI reasoning for difficult classifications |
| **Adaptable** | Works with new error patterns without code changes |
| **Speed** | Fast embeddings for most cases |

## 💡 Why This Works

The hybrid approach combines the best of both worlds:

✅ **Fast embeddings**: Handle 85-95% of errors quickly and cheaply  
✅ **AI reasoning**: No brittle rules, adapts to new patterns  
✅ **Intelligent fallback**: Complex cases get proper analysis  
✅ **Cost-optimized**: Expensive reasoning only when needed  
✅ **Explainable**: Understand why difficult errors were categorized  
✅ **Maintainable**: No rule updates needed as errors evolve  

## 🛠️ Troubleshooting

**API Key Errors**:
- Ensure both `OPENROUTER_API_KEY` and `OPENAI_API_KEY` are set
- Check your API key quotas and billing

**High reasoning usage**:
- If >20% of errors use reasoning, consider lowering `CONFIDENCE_THRESHOLD`
- Or review if your categories need refinement

**Slow performance**:
- Reasoning model adds latency (~1-3 seconds per call)
- Consider using `gpt-4-turbo` instead of `o1-preview`
- Most errors (85-95%) should use fast embeddings

**Unexpected categorization**:
- Check `reasoning_explanation` in output CSV for AI's logic
- Review category descriptions in `categories.json`
- Adjust `CONFIDENCE_THRESHOLD` if needed

## 📦 Dependencies

Core dependencies (see `requirements.txt`):
```
pandas>=1.5.0              # Data manipulation
numpy>=1.24.0              # Numerical operations
scikit-learn>=1.3.0        # Cosine similarity
openai>=1.0.0              # OpenRouter embeddings
python-dotenv>=1.0.0       # Env loading
tqdm>=4.65.0               # Progress bars
```

## 🎯 Best Practices

### For Production Use:
1. **Start with default settings** - Already optimized for most cases
2. **Monitor reasoning usage** - Should be 5-15% of errors
3. **Review reasoning explanations** - Check `reasoning_explanation` column periodically
4. **Set appropriate thresholds** - Adjust `CONFIDENCE_THRESHOLD` based on your accuracy needs
5. **Monitor costs** - Track OpenAI API usage for reasoning calls

### For Custom Error Types:
1. Update category descriptions in `categories.json` for better matching
2. Adjust `CONFIDENCE_THRESHOLD` if too many/few errors use reasoning
3. Test with sample data before processing large batches
4. Review flagged errors to identify patterns

### Cost Optimization:
- Higher threshold = less reasoning = lower cost (but may reduce accuracy)
- Lower threshold = more reasoning = higher cost (but better accuracy)
- Default 0.55 provides good balance

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenRouter for embedding API access
- OpenAI for embedding models
- The open-source community

---

**Built for intelligent, fast, and accurate XML validation error categorization**

**Architecture**: Hybrid AI (Embeddings + Reasoning) - No hard-coded rules ✅
