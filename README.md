# ragasmatrics

# RAGAs Metrics Integration

## 🔍 Objective
This project computes RAGAs metrics (`faithfulness`, `answer_relevancy`, and `context_precision`) for a given LLM log JSON file.

---

## 🚀 Approach

1. **Input Handling**:  
   The `log.json` file is parsed to extract:
   - `context`: Taken from the `system` role in `input`.
   - `query`: Taken from the `user` role in `input`.
   - `answer`: Taken from `expectedOutput` (assistant response).

2. **RAGAs Evaluation**:  
   Using the [RAGAs](https://github.com/explodinggradients/ragas) framework to evaluate the extracted context-query-answer triplets.

3. **Output**:  
   The final output is written to `output.json` with RAGAs scores for each log item:
   ```json
   [
     {
       "id": "item-001",
       "faithfulness": 0.92,
       "answer_relevancy": 0.88,
       "context_precision": 0.95
     },
     ...
   ]
````

---

## 📚 Libraries Used

* `ragas` – for computing evaluation metrics
* `pandas` – for structuring data
* `json` – for reading and writing files
* `tqdm` – for progress tracking
* `langchain`, `langchain_community`, `langchain_core` – for text embedding and document structure
* `openai` – as LLM backend for evaluation (via API key)

---

## 📌 Assumptions & Simplifications

* The `system` role is used as the **context**.
* The `user` role is treated as the **query**.
* The first response in `expectedOutput` is considered the **answer**.
* The API key is assumed to be stored as an environment variable (`OPENAI_API_KEY`).
* Faithfulness metric is **only computed if ground truth is derivable**; otherwise, it is skipped or set to `null`.

---

## 📁 Repository Contents

| File                   | Description                              |
| ---------------------- | ---------------------------------------- |
| `ragas_integration.py` | Python script to compute RAGAs metrics   |
| `log.json`             | Input file with system/user prompts      |
| `output.json`          | Output with computed RAGAs scores        |
| `README.md`            | This README with approach and setup info |

---

## 🛠️ Setup Instructions

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key**:

   * In your terminal or Colab:

     ```python
     import os
     os.environ["OPENAI_API_KEY"] = "your-api-key"
     ```

3. **Run the script**:

   ```bash
   python ragas_integration.py
   ```

4. **Check `output.json`** for results.

---

