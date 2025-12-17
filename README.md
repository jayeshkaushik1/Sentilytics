# Sentilytics 🧠📊
Sentiment Analysis with BERT Embeddings + Classical Machine Learning

Sentilytics is a robust sentiment analysis system that combines the power of BERT-based text representations with classical machine learning models to perform accurate three-way sentiment classification on real-world tweet and review data (`positive`, `neutral`, `negative`).

---

## 🚀 Project Overview

This project demonstrates a practical approach to sentiment analysis by leveraging **BERT as a feature extractor** and then training multiple classical ML models on top of those rich 768-dimensional embeddings. It also includes strong TF-IDF baselines for comparison, enabling you to see exactly how much performance gains come from using modern transformers versus traditional methods.

The complete pipeline includes:
- Data preprocessing with intelligent stopword handling
- Rich exploratory data analysis (EDA) with visualizations
- TF-IDF baseline models for reference
- Production-ready BERT embedding extraction
- Multiple classical ML models trained on embeddings
- Comprehensive evaluation metrics and comparison plots

---

## 📂 Dataset Requirements

The project expects two CSV files in your working directory:

- `train.csv` – Training dataset
- `test.csv` – Test dataset

**File Structure:**
```
text,sentiment,Age of User,Time of Tweet
"Your tweet here",positive,25-34,Morning
"Another tweet",neutral,18-24,Evening
```

Each record contains:
- `text`: The tweet or short review content
- `sentiment`: One of `positive`, `neutral`, or `negative`
- Additional metadata (age group, time of tweet) for deeper analysis

---

## 🧹 Text Preprocessing Pipeline

Sentilytics implements intelligent text preprocessing that maintains semantic meaning while cleaning noise:

- **Lowercasing** for uniformity
- **Smart stopword removal** – preserves important negations (`not`, `no`, `don't`, `isn't`, etc.)
- **URL removal** for cleaner text signals
- **Special character stripping** while keeping meaningful punctuation
- **Whitespace normalization** for consistency

The preprocessing function is applied uniformly to both training and test data, ensuring consistency throughout the pipeline.

---

## 📊 Exploratory Data Analysis (EDA)

The notebook includes comprehensive visualizations:

- **Sentiment distribution** across all classes
- **Sentence length analysis** grouped by sentiment category
- **Demographic breakdowns** (Sentiment vs Age Group, Sentiment vs Time of Tweet)
- **Word frequency analysis** – top 20 words for each sentiment class
- **Beautiful styled plots** with custom color palettes and grid layouts

These visualizations help identify patterns and give you confidence in the data quality.

---

## 🧠 Dual Modeling Approach

### Baseline: TF-IDF + Classical ML

First, TF-IDF features are extracted to establish a strong baseline:

**TF-IDF Configuration:**
- Unigrams and bigrams for capturing phrase-level information
- 5000 feature limit for computational efficiency
- Frequency thresholds to filter noise

**Baseline Models:**
- Logistic Regression with class weight balancing
- Multinomial Naive Bayes
- Linear SVM with balanced weights
- Random Forest with 100 estimators

These models provide clear performance benchmarks.

---

### Main Approach: BERT Embeddings + Classical ML ⭐

The core strength of Sentilytics is combining frozen BERT embeddings with traditional ML models.

**Embedding Extraction:**
- Model: `bert-base-uncased`
- Tokenization via HuggingFace `BertTokenizer`
- Representation: **Mean pooling** of the last hidden state
- Output: 768-dimensional embedding per sentence
- Optimization: Embeddings precomputed using PyTorch DataLoaders for speed

**Models Trained on BERT Embeddings:**
- Decision Tree
- Random Forest (🏆 Best performer)
- Logistic Regression
- K-Nearest Neighbors (k=30)
- Gaussian Naive Bayes

**Random Forest** consistently achieves the highest accuracy and is recommended as the production model.

---

## 📈 Evaluation Metrics

Sentilytics provides comprehensive evaluation:

- **Accuracy** – overall correctness
- **Precision** – how many predicted positives are actually positive
- **Recall** – what percentage of actual positives we catch
- **F1-Score** – harmonic mean of precision and recall
- **Confusion Matrices** – visual breakdown with seaborn heatmaps
- **Comparison Charts** – side-by-side model performance visualization

All metrics are calculated for both individual sentiment classes and weighted averages.

---

## 🎯 Model Performance Highlights

Key results:

- **BERT embeddings improve accuracy** over TF-IDF by capturing semantic meaning better
- **Random Forest on BERT** achieves best overall performance
- **Mean pooling** provides stable, noise-resistant representations
- **Classical ML remains efficient** while maintaining strong performance
- **Clear performance gaps** between different models visible in comparison plots

---

## 💡 Use Cases

Sentilytics excels at:

✅ Classifying product reviews  
✅ Analyzing customer feedback  
✅ Processing social media sentiment  
✅ Content moderation and monitoring  
✅ Brand sentiment tracking  
✅ Real-time opinion analysis  

---

## 🛠️ Tech Stack

**Core Libraries:**
- `Python 3.8+`
- `pandas` – data manipulation and analysis
- `numpy` – numerical operations
- `nltk` – natural language processing toolkit
- `scikit-learn` – classical ML models and metrics
- `pytorch` – tensor operations and data handling
- `transformers` – BERT models from HuggingFace
- `matplotlib` & `seaborn` – professional visualizations

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jayeshkaushik1/Sentilytics.git
cd Sentilytics

# Install dependencies
pip install pandas numpy nltk scikit-learn torch transformers matplotlib seaborn
```

### Running the Analysis

1. Place your `train.csv` and `test.csv` in the project directory
2. Open the Jupyter notebook: `sentiment-analysis.ipynb`
3. Run cells sequentially to:
   - Load and explore your data
   - Preprocess text
   - Generate visualizations
   - Train baseline TF-IDF models
   - Extract BERT embeddings
   - Train and evaluate all classical ML models
   - Compare results

---

## 📋 Example Output

The notebook generates:
- EDA visualizations showing sentiment distribution
- Word frequency bar charts for each sentiment
- Model accuracy comparison bar chart
- Confusion matrices for each model
- Detailed classification reports
- Top-performing model identification

---

## 🏆 Key Advantages

✨ **State-of-the-art representations** using BERT  
✨ **Fast inference** with classical ML models  
✨ **Comprehensive analysis** with multiple visualization types  
✨ **Easy to extend** – add new models or datasets easily  
✨ **Production-ready** – clean, well-documented code  
✨ **Reproducible** – fixed random seeds throughout  
✨ **Interpretable** – confusion matrices and metrics for every model  

---

## 📚 Learning Resources

This project is excellent for understanding:
- How transformers work as feature extractors
- Comparing classical ML vs deep learning approaches
- Text preprocessing best practices
- Model evaluation and comparison techniques
- Building complete ML pipelines

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new datasets
- Implement additional models
- Improve visualizations
- Optimize preprocessing
- Create issue templates

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

**Jayesh Kaushik**
- GitHub: [@jayeshkaushik1](https://github.com/jayeshkaushik1)
- IIIT Bhopal B.Tech Student

---

## ⭐ If you find this project useful, please consider giving it a star!

For questions or suggestions, feel free to open an issue or reach out.
