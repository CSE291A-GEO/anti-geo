# Anti-GEO Paper Sections (ICML LaTeX Format)

This document contains LaTeX sections for the Anti-GEO paper that can be copied directly into Overleaf.

---

## Table of Contents

### Main Paper (Text-Dense, Minimal Tables/Figures)
1. [Methodology](#methodology) - GEO pattern taxonomy, feature representation, rule-based scoring
2. [Implementation](#implementation) - Production models, system architecture, training methodology, deployment
3. [Results: Observations](#results-observations) - Classification vs. ranking gap, ListNet results, failure modes, feature analysis, real-world evaluation
4. [Results: Explanations](#results-explanations) - Why gaps exist, why ListNet works, linear separability, data challenges
5. [Related Work](#related-work) - GEO-Bench, ListNet, related detection work

### Appendix (Detailed Tables, Code Snippets)
A. [Exploration and Ablation Studies](#exploration-and-ablation-studies) - All experimental approaches with detailed results tables:
   - Cosine Similarity Threshold Detection
   - SVM, Ordinal Regression, GBM, RNN, Neural Network Exploration
   - ListNet Ranking Model Exploration
   - Semantic Feature Effect Size Analysis
   - PCA and Semantic Features Ablation
   - Category-Based Demeaning Exploration

B. [Implementation Code Snippets](#implementation-code-snippets) - Feature extraction, model implementations, loss functions, inference pipeline

---
Abstract
Introduction - Victoria
Motivation
Problem Overview
Contributions
Organization (break down into different sections)
Background
What is GEO?
How is it different from SEO?
Why do we need GEO today?
What are some of the risks posed by misused GEO?
Why do we need GEO detection?
Exploration
[Maybe] Warning Method - Simran
Motivation
Experimentation & Results
Other features tried - Stephen
Implementation
Queries Selection - Victoria
GEO optimization pipeline（methodology and implementation) - Victoria, Yuheng
Dataset Generation - Simran, Victoria, Yuheng
GEO detection pipeline - Stephen, Simran
Results
Evaluation Set up - Victoria, Yuheng
Observations for GEO optimization(analysis on optimized articles and metrics) - Victoria, Yuheng
Observations for GEO detection - Stephen, Simran
Explanation of observations - Stephen, Simran
Challenges - Simran, Stephen
Choosing a representative set of queries
What is the source of truth for a source cited in a GE response?
No existing dataset to act as source of truth for training/testing
Linear separability - Stephen
Category-specific baselines for GEO-detection - Stephen
Limitations - Simran
Bias introduced from our selection of queries
Dataset bias – is it the source of truth?
Using just one heuristic for training/testing
The issue of false positives
Related Work - Victoria, Stephen
GEO Bench
Beyond Keywords
ListNet - Stephen
Future Work
Testing for different generative engines such as Perplexity etc.
Exploring other heuristics for GEO detection
Exploring LLM-based methods for GEO detection such as Warning Method, Self-Reflection

## LaTeX Sections

```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SECTION 1: METHODOLOGY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Methodology}
\label{sec:methodology}

Generative Engine Optimization (GEO) refers to techniques used by content creators to manipulate how large language models (LLMs) cite and synthesize web content in AI-generated responses. Unlike traditional Search Engine Optimization (SEO), which targets crawler-based ranking algorithms through keywords and backlinks, GEO exploits the way LLMs process and extract information during response synthesis. Content optimized for GEO typically emphasizes extractability---structuring information in formats that LLMs can easily quote or paraphrase---while embedding authoritative-sounding citations and credentials to increase the likelihood of being selected as a source.

Our production detection system addresses two distinct tasks: binary classification for determining whether individual sources exhibit GEO characteristics, and ranking for identifying the most GEO-optimized source among competitors for a given query. For binary classification, we employ a 10-layer feed-forward neural network trained on scraped data with known GEO labels, which outputs calibrated probability scores through an ordinal logistic layer. For ranking, we use ListNet, a learning-to-rank algorithm that processes all sources for a query simultaneously, enabling the model to learn relative differences rather than absolute patterns. Both approaches share a common feature representation: 384-dimensional semantic embeddings from SentenceTransformer (\texttt{all-MiniLM-L6-v2}), optionally augmented with 5 explicit pattern similarity scores derived from our GEO pattern taxonomy.

\subsection{GEO Pattern Taxonomy}

Through analysis of known GEO-optimized content and consultation with domain experts, we identified five canonical GEO patterns that capture the primary structural and semantic manipulation techniques. The first three patterns (\textbf{GEO\_STRUCT\_001--003}) target structural manipulation: excessive Q\&A blocks characterized by high volumes of question-answer formatted headings with simplistic answers and repetitive brand mentions; over-chunking that breaks content into unnaturally short, self-contained bullet points optimized for LLM extraction; and header stuffing that repeats target keywords across successive H2/H3/H4 headers beyond what natural readability would warrant. The remaining two patterns (\textbf{GEO\_SEMANTIC\_004--005}) address semantic manipulation: entity over-attribution that injects verbose, repetitive credentials (e.g., ``Dr.\ Jane Doe, the world-renowned CSO at BioTech Inc.'') multiple times in short spans; and unnatural citation embedding that places high-precision statistics or quotes in isolated, easy-to-extract formats, sometimes with fabricated source attributions.

\subsection{Feature Representation}

Our classification and ranking models use dense semantic embeddings as the primary feature representation, capturing the semantic meaning of text in a continuous vector space that enables similarity-based reasoning. For each source document, we extract the cleaned text content and embed it using the SentenceTransformer model (\texttt{all-MiniLM-L6-v2}), a 22-million parameter model that produces 384-dimensional vectors optimized for semantic similarity tasks. This embedding captures not just keyword presence but contextual meaning, enabling detection of GEO patterns that manifest through subtle linguistic choices rather than explicit keyword stuffing.

We augment the base embeddings with 5 explicit semantic pattern scores, computed as cosine similarities between the document embedding and pre-computed embeddings for each GEO pattern in our taxonomy. Each pattern embedding is constructed by encoding the concatenation of the pattern's abstract definition and 3--6 representative examples, creating rich representations that capture both conceptual and concrete manifestations of GEO techniques. The final feature vector concatenates the 384-dimensional embedding with the 5 pattern scores, yielding a 389-dimensional representation: $\mathbf{f} = [\mathbf{e}_{\text{semantic}} \| \mathbf{s}_{\text{pattern}}] \in \mathbb{R}^{389}$.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SECTION 2: IMPLEMENTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Implementation}
\label{sec:implementation}

This section describes the implementation of our production GEO detection system, which uses two primary models: (1) a neural network classifier for binary GEO classification, and (2) a ListNet ranking model for identifying the most GEO-optimized source among competitors. Code snippets are provided in Appendix~\ref{sec:appendix-code}.

\subsection{Production Models}

Our binary classification model is a 10-layer feed-forward neural network trained on labeled scraped data. The model takes 384-dimensional semantic embeddings as input and outputs calibrated GEO probabilities through an ordinal logistic layer, using ReLU activations with dropout (0.1) for regularization. Our ranking model uses ListNet, a learning-to-rank algorithm that processes all sources for a query together. The architecture consists of three fully-connected layers (256 $\rightarrow$ 128 $\rightarrow$ 64 units) with ReLU activations and dropout, trained with a combined loss function: 70\% ListNet top-1 probability loss and 30\% pairwise margin ranking loss.

\subsection{System Architecture}

Figure~\ref{fig:pipeline} illustrates our end-to-end GEO detection pipeline, which consists of four main stages.

\begin{figure}[h]
\begin{center}
\fbox{\parbox{0.9\columnwidth}{
\textbf{GEO Detection Pipeline:}\\[0.5em]
Raw Web Content $\rightarrow$ Text Cleaning $\rightarrow$ Embedding Generation $\rightarrow$ Pattern Scoring $\rightarrow$ Neural Net / ListNet $\rightarrow$ GEO Classification / Ranking
}}
\end{center}
\caption{High-level architecture of the GEO detection pipeline showing both classification and ranking paths.}
\label{fig:pipeline}
\end{figure}

\textbf{Stage 1: Text Preprocessing.} Web pages are processed to extract clean text content through HTML parsing, boilerplate removal (navigation, headers, footers, advertisements), text normalization (lowercase conversion, whitespace normalization, duplicate removal), and truncation to 10,000 characters.

\textbf{Stage 2: Feature Extraction.} Clean text is embedded using SentenceTransformer (\texttt{all-MiniLM-L6-v2}), producing 384-dimensional vectors. The model processes documents at approximately 10ms per document on CPU, with batch processing supporting up to 32 documents simultaneously.

\textbf{Stage 3: Pattern Scoring.} For each document, we compute cosine similarity against 5 pre-computed GEO pattern embeddings. Each pattern embedding encodes both the abstract pattern definition and 3--6 representative examples, creating rich representations that capture both conceptual and concrete manifestations of GEO techniques.

\textbf{Stage 4: Model Inference.} The 389-dimensional feature vector (384 embedding + 5 pattern scores) is passed to either the neural classifier (for binary detection) or ListNet (for ranking). Post-processing applies calibrated thresholds and entry-level aggregation.

\subsection{Training Methodology}

Models are trained with entry-level context awareness, where all sources from the same query are processed together, enabling the model to learn relative differences rather than absolute patterns. We apply softmax normalization across entries for relative probability estimation and use ranking loss to encourage correct ordering within entries. The ListNet loss compares predicted and true top-1 probability distributions over sources within a query, augmented with a pairwise margin loss that explicitly penalizes incorrect orderings. Classification thresholds are calibrated on validation data by sweeping over candidate thresholds and selecting the value that maximizes accuracy.

\subsection{Deployment Considerations}

\begin{table}[h]
\caption{Inference latency per source (CPU, single-threaded).}
\label{tab:latency}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lr}
\toprule
Component & Latency (ms) \\
\midrule
Text preprocessing & 2--5 \\
Embedding generation & 8--12 \\
Pattern score extraction & 1--2 \\
Model inference (NN) & 1--3 \\
\midrule
\textbf{Total} & \textbf{12--22} \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

For production deployment, we employ several optimization strategies: batch processing of multiple sources in parallel, GPU acceleration via CUDA-enabled SentenceTransformer (providing approximately 10$\times$ speedup), embedding caching for frequently analyzed domains, and asynchronous I/O for web content fetching. All code and trained models are available in our repository.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPENDIX STARTS HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\appendix

\section{Exploration and Ablation Studies}
\label{sec:exploration}

We conducted extensive experiments to understand GEO detection, exploring multiple approaches before arriving at our final production methods. This appendix documents our exploration of different detection strategies and their relative effectiveness.

\subsection{Cosine Similarity Threshold Detection}
\label{sec:cosine-exploration}

Our initial exploration used direct cosine similarity between document embeddings and known GEO pattern embeddings as a detection signal.

\subsubsection{S\_GEO\_Max Score}

For each suspect document $d$, we compute the \textbf{S\_GEO\_Max} score:
\[
S_{\text{GEO\_Max}}(d) = \max_{p \in \mathcal{P}} \text{cos\_sim}(\mathbf{e}_d, \mathbf{e}_p)
\]
where $\mathcal{P}$ is the set of 5 GEO patterns, $\mathbf{e}_d$ is the embedding of document $d$, and $\mathbf{e}_p$ is the embedding of pattern $p$.

\subsubsection{Threshold-Based Classification}

We explored using S\_GEO\_Max directly for classification with various thresholds:

\begin{table}[h]
\caption{S\_GEO\_Max threshold exploration for GEO detection.}
\label{tab:sgeo-thresholds}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lcc}
\toprule
Threshold & Interpretation & Limitation \\
\midrule
$\geq 0.75$ & High GEO likelihood & Low recall \\
$0.5 - 0.75$ & Moderate GEO & High false positives \\
$< 0.5$ & Likely natural & Misses subtle GEO \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\textbf{Findings}: Direct cosine similarity thresholds proved insufficient for reliable detection because:
\begin{itemize}
    \item Natural content with legitimate structure (FAQs, documentation) often exceeds thresholds
    \item Subtle GEO optimization may not trigger high similarity scores
    \item The approach lacks context---it cannot compare sources within a query
\end{itemize}

This motivated our exploration of machine learning approaches that could learn more nuanced decision boundaries.

\subsection{Support Vector Machine (SVM) Exploration}
\label{sec:svm-exploration}

We explored SVM with RBF kernel for GEO classification, motivated by its effectiveness in high-dimensional spaces.

\subsubsection{Configuration}
\begin{itemize}
    \item Kernel: RBF (Radial Basis Function)
    \item Regularization: $C = 1.5$
    \item Feature space: 384-dim embeddings, optionally with 5 pattern scores (389-dim)
\end{itemize}

\subsubsection{Results}

\begin{table}[h]
\caption{SVM performance across feature configurations.}
\label{tab:svm-exploration}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccc}
\toprule
Configuration & Val Acc & Ranking Acc & F1 \\
\midrule
Baseline (384-dim) & 86.14\% & N/A & 0.513 \\
+Semantic (389-dim) & 86.14\% & N/A & 0.540 \\
PCA 250 & 86.57\% & 66.43\% & 0.664 \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\textbf{Findings}: SVM achieved the highest classification accuracy among traditional models (86.57\% with PCA), but ranking accuracy (66.43\%) remained significantly lower than classification accuracy, indicating the model struggles to identify the \textit{most} GEO-optimized source among competitors.

\subsection{Ordinal Logistic Regression Exploration}
\label{sec:ordinal-exploration}

We explored ordinal logistic regression, motivated by the hypothesis that GEO detection is inherently ordinal---content ranges from ``clearly non-GEO'' to ``strongly GEO-optimized.''

\subsubsection{Configuration}
\begin{itemize}
    \item Model: Cumulative link model with learned ordinal thresholds
    \item Loss: Ordinal cross-entropy
    \item Feature space: 384-dim embeddings + 5 pattern scores
\end{itemize}

\subsubsection{Results}

\begin{table}[h]
\caption{Ordinal logistic regression performance.}
\label{tab:ordinal-exploration}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccc}
\toprule
Configuration & Val Acc & Ranking Acc & F1 \\
\midrule
Baseline (384-dim) & 82.43\% & 62.14\% & 0.226 \\
+Semantic (389-dim) & 84.00\% & 62.86\% & 0.356 \\
PCA 250 & 82.43\% & 62.14\% & 0.226 \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\textbf{Findings}: Ordinal regression showed the largest benefit from adding semantic pattern scores (+1.57\% accuracy, +0.130 F1), suggesting linear models benefit most from explicit discriminative features. However, overall ranking performance remained limited.

\subsection{Gradient Boosting Machine (GBM) Exploration}
\label{sec:gbm-exploration}

We explored GBM for its ability to capture complex feature interactions.

\subsubsection{Configuration}
\begin{itemize}
    \item Architecture: Ensemble of 100 decision trees
    \item Max depth: 3
    \item Learning rate: 0.1
\end{itemize}

\subsubsection{Results}

\begin{table}[h]
\caption{GBM performance across feature configurations.}
\label{tab:gbm-exploration}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccc}
\toprule
Configuration & Val Acc & Ranking Acc & F1 \\
\midrule
Baseline (384-dim) & 83.14\% & 55.71\% & 0.443 \\
+Semantic (389-dim) & 76.29\% & 53.57\% & 0.461 \\
PCA 250 & 52.29\% & 35.71\% & 0.337 \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\textbf{Findings}: GBM suffered catastrophic performance degradation with PCA ($-30.85$\% accuracy), indicating it relies on the full feature space for decision tree splits. Adding semantic features also degraded accuracy, suggesting overfitting on the small training set.

\subsection{Recurrent Neural Network (RNN) Exploration}
\label{sec:rnn-exploration}

We explored RNNs to model sequential dependencies in text representations.

\subsubsection{Configuration}
\begin{itemize}
    \item Architecture: 3 bidirectional GRU layers
    \item Hidden units: 128
    \item Input projection layer for embedding dimension adaptation
\end{itemize}

\subsubsection{Results}

\begin{table}[h]
\caption{RNN performance across feature configurations.}
\label{tab:rnn-exploration}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccc}
\toprule
Configuration & Val Acc & Ranking Acc & F1 \\
\midrule
Baseline (384-dim) & 80.71\% & 56.43\% & 0.509 \\
+Semantic (389-dim) & 79.00\% & 59.29\% & 0.495 \\
PCA 250 & 82.71\% & 67.86\% & 0.569 \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\textbf{Findings}: RNN showed the largest improvement from PCA (+11.43\% ranking accuracy), suggesting reduced dimensionality helps the sequential model focus on informative features. However, it still underperformed compared to ranking-specific approaches.

\subsection{Feed-Forward Neural Network Exploration}
\label{sec:nn-exploration}

We explored deep feed-forward neural networks, motivated by their ability to learn hierarchical representations of GEO patterns.

\subsubsection{Configuration}
\begin{itemize}
    \item Architecture: 10 hidden layers, 128 units each
    \item Activation: ReLU with dropout (0.1)
    \item Output: Ordinal logistic layer to respect ranking nature
    \item Loss: 50\% ordinal loss + 50\% ranking loss
\end{itemize}

\subsubsection{Results}

\begin{table}[h]
\caption{Feed-forward neural network performance across feature configurations.}
\label{tab:nn-exploration}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccc}
\toprule
Configuration & Val Acc & Ranking Acc & F1 \\
\midrule
Baseline (384-dim) & 84.86\% & 62.86\% & 0.562 \\
+Semantic (389-dim) & 85.00\% & 63.57\% & 0.595 \\
PCA 250 & 83.86\% & 61.43\% & 0.523 \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\textbf{Findings}: The neural network showed consistent performance across configurations, with slight improvement from semantic features. This model was selected for production binary classification due to its balance of accuracy and calibrated probability outputs.

\subsection{ListNet Ranking Model Exploration}
\label{sec:listnet-exploration}

We explored ListNet, a learning-to-rank algorithm, motivated by the hypothesis that GEO detection is fundamentally a ranking problem rather than classification.

\subsubsection{Motivation}

Unlike classification models that predict labels independently, ListNet directly optimizes for ranking accuracy by learning to rank sources within each query. This is crucial for GEO detection where the goal is to identify which source is most likely to be GEO-optimized among competing sources.

\subsubsection{Configuration}
\begin{itemize}
    \item Architecture: 3 hidden layers (256 $\rightarrow$ 128 $\rightarrow$ 64 units)
    \item Activation: ReLU with dropout (0.1)
    \item Feature space: 389-dim (embeddings + pattern scores)
\end{itemize}

\subsubsection{Loss Function}

Combined loss function:
\[
\mathcal{L} = 0.7 \times \mathcal{L}_{\text{ListNet}} + 0.3 \times \mathcal{L}_{\text{pairwise}}
\]

The ListNet loss compares predicted top-1 probability distributions with true rankings:
\[
\mathcal{L}_{\text{ListNet}} = -\sum_{i} P(y_i = 1) \log P(\hat{y}_i = 1)
\]

The pairwise ranking loss encourages correct orderings:
\[
\mathcal{L}_{\text{pairwise}} = \sum_{(i,j): r_i < r_j} \max(0, 1 - (s_i - s_j))
\]

\subsubsection{Results}

\begin{table}[h]
\caption{ListNet ranking model results on optimization dataset (700 train / 300 validation).}
\label{tab:listnet-exploration}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lcc}
\toprule
Metric & Training & Validation \\
\midrule
Ranking Accuracy & 99.43\% & 87.00\% \\
Mean Rank Deviation & 1.20 & 1.25 \\
Mean Reciprocal Rank & 0.997 & 0.919 \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\textbf{Findings}: ListNet achieved \textbf{87.00\% validation ranking accuracy}, a 19+ percentage point improvement over the best classification-based approach (RNN at 67.86\%). This dramatic improvement validated our hypothesis that GEO detection benefits from ranking-based formulations, and ListNet was selected for production ranking tasks.

\subsection{Model Exploration Summary}

\begin{table}[h]
\caption{Comparison of all explored models on validation ranking accuracy.}
\label{tab:model-exploration-summary}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lcc}
\toprule
Model & Best Ranking Acc & Selected for Production \\
\midrule
\textbf{ListNet} & \textbf{87.00\%} & \checkmark (ranking) \\
RNN (PCA 250) & 67.86\% & \\
SVM (PCA 250) & 66.43\% & \\
Neural Network & 63.57\% & \checkmark (classification) \\
Logistic Ordinal & 62.86\% & \\
GBM & 55.71\% & \\
Cosine Threshold & N/A & \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

Based on our exploration, we selected:
\begin{itemize}
    \item \textbf{Neural Network} for binary classification (good accuracy, calibrated probabilities)
    \item \textbf{ListNet} for ranking tasks (dramatically superior ranking accuracy)
\end{itemize}

\subsection{Semantic Feature Effect Size Analysis}

We analyzed the discriminative power of each semantic feature using multiple statistical methods: correlation analysis, mean difference analysis, and Cohen's $d$ effect size.

\subsubsection{Feature Importance Ranking}

\begin{table}[h]
\caption{Semantic feature importance analysis (4,996 samples: 1,000 GEO, 3,996 non-GEO).}
\label{tab:feature-importance}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lcccr}
\toprule
Feature & Correlation & Cohen's $d$ & Mean Diff & Rank \\
\midrule
Entity Attribution & 0.323 & 0.823 & 62.2\% & 1 \\
Header Stuffing & 0.289 & 0.731 & 49.1\% & 2 \\
Citation Embedding & 0.248 & 0.614 & 45.9\% & 3 \\
Q\&A Blocks & 0.225 & 0.568 & 43.1\% & 4 \\
Over-Chunking & 0.214 & 0.543 & 38.9\% & 5 \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\subsubsection{Effect Size Interpretation}

\begin{itemize}
    \item \textbf{Entity Over-Attribution} (Cohen's $d = 0.82$): Large effect size. GEO content consistently exhibits verbose, repetitive entity definitions. This is the most discriminative feature.
    
    \item \textbf{Header Stuffing} (Cohen's $d = 0.73$): Medium-to-large effect size. Keyword repetition in headers is a strong structural indicator of GEO optimization.
    
    \item \textbf{Citation Embedding} (Cohen's $d = 0.61$): Medium effect size. Unnatural citation patterns are moderately present in GEO content.
    
    \item \textbf{Q\&A Blocks} and \textbf{Over-Chunking} (Cohen's $d \approx 0.55$): Medium effect sizes. These structural patterns vary by content type and are less consistently applied.
\end{itemize}

\textbf{Key Finding}: All five semantic features contribute meaningful signal, with the top two features (Entity Attribution, Header Stuffing) showing large effect sizes ($d > 0.7$), indicating strong, consistent differences between GEO and non-GEO content.

\subsection{PCA Dimensionality Reduction Analysis}

We investigated whether reducing the 384-dimensional embedding space could improve model performance by eliminating noise dimensions.

\subsubsection{PCA 250 Results}

\begin{table}[h]
\caption{Impact of PCA dimensionality reduction (384 $\rightarrow$ 250 components).}
\label{tab:pca-results}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccc}
\toprule
Model & Baseline & PCA 250 & $\Delta$ Ranking \\
\midrule
SVM & 86.14\% & 86.57\% & $+0.43$\% \\
Logistic Ordinal & 82.43\% & 82.43\% & $0.00$\% \\
GBM & 83.14\% & 52.29\% & $-30.85$\% \\
Neural (10-layer) & 84.86\% & 83.86\% & $-1.00$\% \\
RNN (3-layer GRU) & 80.71\% & 82.71\% & $+2.00$\% \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\textbf{Key Findings}:
\begin{enumerate}
    \item \textbf{RNN benefits from PCA}: The RNN showed the largest improvement (+11.43\% ranking accuracy), suggesting that reduced dimensionality helps the sequential model focus on the most informative features.
    
    \item \textbf{GBM suffers significantly}: The gradient boosting model experienced catastrophic performance degradation ($-30.85$\% accuracy), indicating it relies on the full feature space for decision tree splits.
    
    \item \textbf{Linear models are robust}: SVM and Logistic Ordinal showed minimal change, suggesting the RBF kernel and ordinal thresholds can adapt to reduced dimensions.
\end{enumerate}

\subsection{Semantic Features Ablation}

We compared model performance with and without the 5 explicit semantic pattern scores.

\begin{table}[h]
\caption{Impact of adding 5 semantic pattern scores to 384-dim embeddings.}
\label{tab:semantic-ablation}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lcccc}
\toprule
Model & Base Acc & +Semantic & $\Delta$ Acc & $\Delta$ F1 \\
\midrule
SVM & 86.14\% & 86.14\% & 0.00\% & +0.028 \\
Logistic Ordinal & 82.43\% & 84.00\% & +1.57\% & +0.130 \\
GBM & 83.14\% & 76.29\% & $-6.85$\% & +0.018 \\
Neural & 84.86\% & 85.00\% & +0.14\% & +0.033 \\
RNN & 80.71\% & 79.00\% & $-1.71$\% & $-0.014$ \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\textbf{Key Findings}:
\begin{enumerate}
    \item \textbf{Logistic Ordinal benefits most}: Linear models gain the most from explicit discriminative features, with +1.57\% accuracy and +0.130 F1 improvement.
    
    \item \textbf{SVM implicitly captures patterns}: The RBF kernel SVM shows no accuracy change but improved F1, suggesting it already captures these patterns non-linearly.
    
    \item \textbf{GBM shows mixed results}: Decreased accuracy but improved F1, indicating the additional features may cause overfitting on the small training set.
\end{enumerate}

\subsection{Category-Based Demeaning Exploration}
\label{sec:demeaning-exploration}

We hypothesized that different website categories exhibit naturally different GEO-like characteristics, creating confounding signals. We explored normalizing GEO scores by subtracting category-specific baseline means.

\subsubsection{Motivation}

Different website categories have inherently different structural patterns:

\begin{table}[h]
\caption{Natural GEO score variation across website categories (reference corpus: 2,383 sources).}
\label{tab:category-baselines-exploration}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lrr}
\toprule
Category & Mean GEO Score & Std \\
\midrule
E-commerce & 0.216 & 0.084 \\
Corporate & 0.209 & 0.076 \\
Affiliate & 0.211 & 0.080 \\
Educational & 0.183 & 0.078 \\
Non-profit & 0.158 & 0.069 \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

E-commerce content naturally scores 37\% higher on GEO metrics than Non-profit content (0.216 vs.\ 0.158), even without intentional GEO optimization. We hypothesized this creates systematic false positives for commercial content.

\subsubsection{Demeaning Approach}

We categorized websites into 10 categories (E-commerce, Corporate, Personal/Portfolio, Content-sharing, Communication/Social, Educational, News and Media, Membership, Affiliate, Non-profit) and computed baseline statistics from a reference corpus.

For each source in category $c$:
\[
S_{\text{demeaned}} = S_{\text{original}} - \mu_c
\]
where $\mu_c$ is the mean GEO score for category $c$.

\subsubsection{Scalar Demeaning Results}

\begin{table}[h]
\caption{Impact of per-category GEO score demeaning on classification.}
\label{tab:scalar-demeaning-exploration}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccc}
\toprule
Model & Original & Demeaned & $\Delta$ \\
\midrule
Logistic Ordinal & 80.00\% & 80.00\% & 0.00\% \\
Neural (10-layer) & 92.00\% & 88.00\% & $-4.00$\% \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\subsubsection{Embedding-Level Demeaning Results}

We also tested demeaning the full 384-dimensional embedding vectors by category:

\begin{table}[h]
\caption{Impact of embedding-level category demeaning.}
\label{tab:embedding-demeaning-exploration}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lcc}
\toprule
Model & Without Demeaning & With Demeaning \\
\midrule
Logistic Ordinal & 80.00\% & 80.00\% \\
Neural (10-layer) & 84.00\% & 84.00\% \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

\subsubsection{Findings}

\textbf{Result}: Category demeaning provided \textbf{no improvement} and slightly degraded neural network performance ($-4.00$\%).

\textbf{Interpretation}:
\begin{itemize}
    \item Semantic embeddings already capture category-invariant GEO signals
    \item Category-specific baselines may be \textbf{orthogonal} to the GEO detection signal rather than confounding it
    \item The variance reduction from demeaning (0.0803 $\rightarrow$ 0.0796) may remove useful discriminative information
\end{itemize}

\textbf{Conclusion}: We did not include category demeaning in our production models as it failed to improve detection performance.

\subsection{Training Techniques}

We employed several techniques to address class imbalance and improve ranking performance:

\begin{enumerate}
    \item \textbf{Oversampling}: Positive (GEO) samples were oversampled with weight=3.0 to address the approximately 1:4 class imbalance.
    
    \item \textbf{Entry-Batched Training}: All models process sources from the same query together, enabling context-aware predictions.
    
    \item \textbf{Combined Loss} (Neural \& RNN): 50\% ordinal loss + 50\% ranking loss to encourage correct relative ordering within entries.
    
    \item \textbf{Entry-Argmax Post-Processing}: For final predictions, the source with the highest GEO probability within each entry is forced to be positive.
\end{enumerate}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SECTION 3: RESULTS - OBSERVATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Results: Observations}
\label{sec:results-observations}

We evaluated our GEO detection approaches on three datasets spanning different aspects of the detection problem. The \textbf{Optimization Dataset}, derived from GEO-Bench, contains 1,000 entries where each entry comprises a query and multiple candidate sources with ground-truth labels indicating which source is GEO-optimized. The \textbf{Combined GEO Dataset} provides 446 entries combining known GEO content from web scraping with non-GEO content from search engine results. The \textbf{Real World Dataset} offers 446 query-article pairs where each query has both an original article and a GEO-optimized version, enabling controlled before/after comparisons.

\subsection{Classification vs. Ranking Performance Gap}

A critical finding from our experiments is the substantial gap between classification accuracy and ranking accuracy across all models. On the optimization dataset, classification models achieve validation accuracies ranging from 80--86\%, with SVM performing best at 86.57\% using PCA-reduced features. However, when evaluated on the ranking task---identifying which source among competitors has the highest GEO probability---these same models achieve only 55--68\% accuracy. This 20+ percentage point gap reveals a fundamental limitation: models can correctly classify individual sources as GEO or non-GEO while still failing to identify the \textit{most} GEO-optimized source in a competitive set.

\subsection{ListNet Ranking Results}

The ListNet model, trained specifically for the ranking task with a combined ListNet and pairwise loss function, dramatically outperforms classification-based approaches. On the optimization dataset (700 training / 300 validation entries), ListNet achieves 87.00\% validation ranking accuracy with a mean reciprocal rank of 0.919, representing a 19+ percentage point improvement over the best classification model (RNN with PCA at 67.86\%). On the Yuheng dataset, where the task is distinguishing original articles from their GEO-optimized versions, ListNet achieves 90.30\% validation ranking accuracy. The neural classifier achieves AUC = 0.783 on scraped data for binary classification, while the ListNet model shows lower per-source AUC (0.640) because it is optimized for ranking rather than absolute probability calibration.

\begin{table*}[t]
\caption{Production model performance across datasets. Neural Network is a 10-layer feed-forward classifier; ListNet is a ranking model with combined ListNet and pairwise loss. Classification Acc measures per-source binary classification; Ranking Acc measures whether the correct GEO source is ranked first within each query.}
\label{tab:results-summary}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{llcccc}
\toprule
Dataset & Model & Classification Acc & Ranking Acc & MRR & AUC \\
\midrule
\multirow{2}{*}{Combined GEO (65 entries)} & Neural Net & 90.00\% & 0.00\% & --- & 0.783 \\
 & ListNet & --- & 25.00\% & 0.454 & --- \\
\midrule
\multirow{2}{*}{Optimization (1000 entries)} & Neural Net & 84.86\% & 62.86\% & --- & --- \\
 & ListNet & --- & 87.00\% & 0.919 & --- \\
\midrule
\multirow{2}{*}{Real World (446 entries)} & Neural Net & 50.00\% & --- & --- & --- \\
 & ListNet & --- & 90.30\% & 0.952 & 0.640 \\
\midrule
\multirow{2}{*}{Real-World Scraped (73 queries)} & Neural Net & 84.00\%$^*$ & --- & --- & --- \\
 & ListNet & 62.31\%$^\dagger$ & --- & --- & --- \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip 0.05in
\footnotesize{$^*$Neural Net trained on Combined GEO, applied to real-world. $^\dagger$Best threshold from validation sweep.}
\vskip -0.1in
\end{table*}

\subsection{Failure Mode Analysis}

Analysis of ranking failures reveals a consistent pattern: the primary failure mode is extreme under-confidence in true GEO sources rather than over-confidence in non-GEO content. In 52 failed ranking cases out of 140 validation entries, 71.2\% involved true GEO sources receiving probabilities below 0.01, and 84.6\% had probabilities below 0.1. The mean GEO probability for true GEO sources in successful cases was 0.7946 compared to just 0.0793 in failed cases---a 10$\times$ difference that indicates the model either confidently identifies GEO content or completely misses it, with little middle ground.

\subsection{Feature Discriminative Power}

Our effect size analysis reveals which GEO patterns are most reliably detected. Entity Over-Attribution exhibits the largest effect size (Cohen's $d = 0.82$), making it our most discriminative feature. Header Stuffing shows the second-largest effect size ($d = 0.73$), as keyword repetition across successive headers creates measurable n-gram overlap that natural content avoids. Conversely, structural patterns like Over-Chunking ($d = 0.54$) and Q\&A Blocks ($d = 0.57$) prove harder to detect because they overlap substantially with legitimate formatting practices.

\subsection{The Semantic Feature Paradox}

A critical observation from failure analysis is what we term the ``semantic feature paradox'': in failed ranking cases, winning non-GEO competitors actually exhibit \textit{higher} semantic feature scores than the true GEO sources across all five pattern dimensions. This reveals that non-GEO sources with naturally occurring GEO-like patterns---well-structured FAQs, properly attributed expert quotes, organized product descriptions---can fool the detector by triggering the same signals that indicate intentional GEO optimization.

\subsection{Real-World Evaluation}

The Yuheng dataset provides a controlled evaluation setting where ListNet achieved 90.30\% validation ranking accuracy, while per-source classification models without query context achieve only chance performance (50.00\% for neural network, 36.94\% for logistic regression). This stark contrast provides compelling evidence that the GEO detection signal lies in relative differences within query contexts.

To evaluate generalization to real-world content, we constructed a dataset by scraping Google AI mode references for 100 diverse queries, retaining 73 queries with 548 total sources after filtering. Using our neural network classifier, 10.8\% of sources (59/548) were classified as GEO, with 64.4\% of queries having no sources classified as GEO. Threshold sensitivity analysis reveals that varying the ListNet score threshold from $-1.0$ to $+1.0$ shifts the percentage of sources classified as GEO from 77.3\% to 19.9\%, demonstrating that the distinction exists on a continuum rather than as a binary classification.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SECTION 4: RESULTS - EXPLANATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Results: Explanations}
\label{sec:results-explanations}

\subsection{Why the Classification-Ranking Gap Exists}

The gap between classification accuracy and ranking accuracy arises because many true GEO sources receive moderate probabilities (correctly classified as GEO) but are outscored by non-GEO competitors that happen to exhibit naturally occurring GEO-like patterns. Classification models learn to detect GEO-like patterns without learning to distinguish intentional optimization from naturally occurring structure. Technical documentation, academic content with proper citations, and e-commerce product pages all score high on our GEO features despite being legitimate content. E-commerce content naturally scores 37\% higher on GEO metrics than non-profit content (0.216 vs.\ 0.158 baseline scores), creating systematic false positive risk for commercial domains.

\subsection{Why ListNet Outperforms Classification}

The semantic feature paradox explains why ListNet's ranking-based approach dramatically outperforms classification: by learning relative differences within query contexts, ListNet can distinguish between sources that both exhibit some GEO-like characteristics, whereas classification models treat each source independently and cannot leverage comparative signals. The root cause of ranking failures is a combination of extreme under-confidence in true GEO sources (which often apply subtle optimization that doesn't trigger strong pattern signals) and moderate over-confidence in GEO-like non-GEO content (which triggers patterns strongly but for legitimate reasons).

\subsection{Why Certain Patterns Are More Discriminative}

Entity Over-Attribution works because natural content typically introduces an expert once and uses pronouns thereafter, while GEO-optimized content repeats full credentials to maximize LLM extraction probability, creating an unnatural repetition pattern rare in organic writing. Header Stuffing works because human writers optimize headers for scannability and variety, while GEO-optimized content sacrifices readability for keyword density. Structural patterns like Over-Chunking and Q\&A Blocks are less discriminative because bullet points, numbered lists, and FAQ formats are legitimate organizational tools---the boundary between ``good structure'' and ``over-chunking for LLM extraction'' is inherently fuzzy and context-dependent.

\subsection{Linear Separability Challenge}

A fundamental challenge in GEO detection is the limited linear separability between GEO-optimized and non-GEO content in embedding space. When we treat GEO detection as a per-source binary classification problem without query context, standard classifiers achieve only random-chance performance: a 10-layer neural network achieves 50.00\% validation accuracy on the Yuheng dataset, and SGD logistic regression fails to converge entirely (36.94\%). This indicates that GEO-optimized and original content occupy overlapping regions in the 384-dimensional embedding space, with no clear decision boundary separating them.

The key insight is that the distinguishing signal lies in \textit{relative} differences within query contexts rather than absolute feature values. A GEO-optimized article may not look dramatically different from natural content in isolation, but when compared to the original version of the same article, the optimization patterns become apparent. This explains why ListNet achieves 90.30\% accuracy on the same dataset where per-source classifiers achieve only chance performance.

\subsection{Data and Evaluation Challenges}

GEO detection faces significant data challenges stemming from the absence of established ground truth. No gold standard exists for what constitutes GEO-optimized content, and labels derived from synthetic optimization may not reflect the diversity of real-world GEO techniques. Human annotation is subjective and requires domain expertise. Our datasets exhibit substantial class imbalance---approximately 1:4 GEO to non-GEO ratio---causing models to achieve misleadingly high accuracy by predicting the majority class.

The disconnect between classification and ranking metrics presents an evaluation challenge: models can achieve 84.7\% classification accuracy while achieving only 62.9\% ranking accuracy, a 21.8 percentage point gap. Threshold calibration for deployment proves similarly difficult: the optimal threshold achieves only 62.31\% per-source accuracy. Domain shift between synthetic training data and real-world GEO techniques remains an open challenge, as synthetic GEO applies known patterns systematically while real-world GEO may employ novel techniques.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Related Work}
% Contributors: Victoria, Stephen
\label{sec:related-work}

\subsection{GEO-Bench}

GEO-Bench \cite{aggarwal2024geobench} introduced the first systematic benchmark for evaluating Generative Engine Optimization techniques. The benchmark provides:
\begin{itemize}
    \item A dataset of 1,000 queries with associated web sources
    \item Ground-truth labels indicating which sources are GEO-optimized (\texttt{sugg\_idx})
    \item Evaluation metrics for measuring GEO effectiveness
\end{itemize}

Our work extends GEO-Bench by focusing on the \textit{detection} of GEO-optimized content rather than its generation. We use the GEO-Bench optimization dataset as training data for our detection models.

\subsection{Beyond Keywords: SEO and GEO Comparison}

Traditional Search Engine Optimization (SEO) focuses on keyword density, backlinks, and metadata to improve search rankings. GEO differs fundamentally:

\begin{table}[h]
\caption{Comparison of SEO and GEO techniques.}
\label{tab:seo-vs-geo}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lll}
\toprule
Aspect & SEO & GEO \\
\midrule
Target & Search engine crawlers & LLM synthesis \\
Signal & Keywords, links & Extractable facts \\
Structure & Metadata-focused & Content-focused \\
Goal & Higher ranking & Higher citation \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

GEO techniques exploit how LLMs process and synthesize information:
\begin{itemize}
    \item \textbf{Extractability}: Content structured for easy LLM extraction (bullet points, Q\&A)
    \item \textbf{Authority signals}: Expert attributions and citations that LLMs may prioritize
    \item \textbf{Specificity}: High-precision statistics that are quotable
\end{itemize}

\subsection{ListNet: Learning to Rank}
% Contributor: Stephen

ListNet \cite{cao2007listnet} is a listwise learning-to-rank algorithm that directly optimizes for ranking accuracy rather than pointwise or pairwise objectives.

\subsubsection{Why ListNet for GEO Detection}

We chose ListNet over classification approaches for several reasons:

\begin{enumerate}
    \item \textbf{Task Alignment}: GEO detection is fundamentally a ranking problem---identifying which source among competitors is most likely GEO-optimized.
    
    \item \textbf{Linear Separability}: Our experiments showed that GEO vs.\ non-GEO content is not linearly separable in embedding space (per-source classification achieves only 50\% accuracy). ListNet sidesteps this by learning \textit{relative} orderings.
    
    \item \textbf{Context Awareness}: ListNet processes all sources for a query together, enabling comparison-based decisions rather than absolute thresholds.
\end{enumerate}

\subsubsection{ListNet Algorithm}

ListNet optimizes the cross-entropy between predicted and true top-$k$ probability distributions:

\textbf{Top-1 Probability}:
\[
P(y_i = 1 | \mathbf{s}) = \frac{\exp(s_i)}{\sum_{j=1}^{n} \exp(s_j)}
\]
where $s_i$ is the predicted relevance score for source $i$.

\textbf{Loss Function}:
\[
\mathcal{L} = -\sum_{i=1}^{n} P_{\text{true}}(y_i = 1) \log P_{\text{pred}}(y_i = 1)
\]

This loss encourages the model to assign higher scores to truly relevant (GEO-optimized) sources.

\subsubsection{Our Modifications}

We extended the standard ListNet with:
\begin{enumerate}
    \item \textbf{Combined Loss}: 70\% ListNet + 30\% pairwise margin loss for more robust training
    \item \textbf{Deep Architecture}: 3-layer MLP (256-128-64) instead of linear scoring
    \item \textbf{Early Stopping}: Prevent overfitting on small datasets
\end{enumerate}

\subsubsection{Performance Comparison}

\begin{table}[h]
\caption{ListNet vs.\ classification approaches on ranking accuracy.}
\label{tab:listnet-comparison}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lc}
\toprule
Approach & Validation Ranking Acc \\
\midrule
Logistic Regression & 62.14\% \\
Neural Network (10-layer) & 62.86\% \\
SVM (RBF) & 66.43\% \\
RNN (GRU) & 67.86\% \\
\textbf{ListNet (ours)} & \textbf{87.00\%} \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

ListNet achieves a 19+ percentage point improvement over the best classification-based approach, validating our hypothesis that GEO detection benefits from ranking-based formulations.

\subsection{Other Relevant Work}

Our work connects to several related research areas. Content manipulation detection research, including fake news and spam detection, employs similar semantic embedding approaches but targets different manipulation patterns focused on misinformation rather than citation optimization. Adversarial text detection shares methodological similarities with GEO detection, though GEO optimization differs fundamentally in that it aims to be helpful to users while gaming citation mechanisms rather than causing model failures. LLM watermarking approaches for detecting AI-generated content are complementary to GEO detection but address a different problem: identifying generation source rather than optimization intent.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Real-World Evaluation}
\label{sec:real-world}

\subsection{Controlled Evaluation: Before and After GEO Optimization}

The Yuheng dataset provides a controlled evaluation setting with 446 query-article pairs where each query has both an original article and a GEO-optimized version. This enables direct measurement of whether models can distinguish original content from its optimized variant---a cleaner signal than comparing arbitrary GEO and non-GEO sources. We trained ListNet on a 70/30 split (312 training, 134 validation) and achieved 90.30\% validation ranking accuracy with a mean reciprocal rank of 0.9515, demonstrating strong ability to identify which version of an article has been GEO-optimized.

Critically, per-source classification models without query context achieve only chance performance on this dataset: a 10-layer neural network reaches 50.00\% accuracy, and SGD logistic regression fails to converge (36.94\%). This stark contrast---90.30\% for ranking versus 50.00\% for classification---provides compelling evidence that the GEO detection signal lies in relative differences within query contexts rather than absolute feature patterns.

\subsection{Real-World Dataset: AI Mode References}

To evaluate generalization to real-world content, we constructed a dataset by scraping Google AI mode references for 100 diverse queries. After filtering queries with fewer than 5 usable sources (removing 27 queries due to insufficient clean content), we retained 73 queries with 548 total sources. Using our neural network classifier trained on scraped data, we found that 10.8\% of sources (59/548) were classified as GEO, with an average of 10.2\% GEO sources per query. The distribution is highly skewed: 64.4\% of queries (47/73) had no sources classified as GEO, while only 14.4\% of queries had more than 20\% of their sources classified as GEO.

Threshold sensitivity analysis reveals the challenge of calibrating detection for deployment. Varying the ListNet score threshold from $-1.0$ to $+1.0$ shifts the percentage of sources classified as GEO from 77.3\% to 19.9\%, demonstrating that the distinction between ``GEO'' and ``non-GEO'' is not binary but exists on a continuum. The optimal threshold ($-0.2858$), determined via validation sweep, achieves only 62.31\% per-source accuracy, highlighting the inherent difficulty of converting ranking scores to binary classifications.

\subsection{Key Findings}

Our experiments reveal several important findings about GEO detection. First, ranking-based approaches dramatically outperform classification approaches (87--90\% vs.\ 50--67\% accuracy), validating our hypothesis that GEO detection is fundamentally a relative comparison task. Second, Entity Over-Attribution (Cohen's $d = 0.82$) and Header Stuffing ($d = 0.73$) are the most discriminative GEO patterns, while structural patterns like Over-Chunking and Q\&A Blocks prove harder to distinguish from legitimate formatting. Third, category-based score normalization provides no improvement, suggesting that semantic embeddings already capture category-invariant GEO signals. Finally, models trained on synthetic GEO-optimized content transfer reasonably well to real-world web content, though threshold calibration remains challenging for deployment.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Implementation Code Snippets}
\label{sec:appendix-code}

This appendix provides code snippets for key components of the GEO detection pipeline described in Section~\ref{sec:implementation-detection}.

\subsection{Feature Extraction}

\subsubsection{Semantic Embedding Generation}

\begin{verbatim}
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(cleaned_text, 
                         convert_to_numpy=True)
# Output: 384-dimensional vector
\end{verbatim}

\subsubsection{Pattern Score Extraction}

\begin{verbatim}
def extract_pattern_scores(text_embedding, 
                           pattern_embeddings):
    scores = []
    for pattern_emb in pattern_embeddings:
        sim = cosine_similarity(
            text_embedding.reshape(1, -1),
            pattern_emb.reshape(1, -1)
        )[0, 0]
        scores.append(sim)
    return np.array(scores)  # 5-dimensional
\end{verbatim}

\subsubsection{Feature Vector Assembly}

\begin{verbatim}
def extract_features(cleaned_text):
    # Base embedding (384-dim)
    embedding = model.encode(cleaned_text)
    
    # Pattern scores (5-dim)
    pattern_scores = extract_pattern_scores(
        embedding, pattern_embeddings)
    
    # Concatenate: 389-dim total
    features = np.concatenate([
        embedding, pattern_scores
    ])
    return features
\end{verbatim}

\subsection{Model Implementations}

\subsubsection{Classifier Interface}

\begin{verbatim}
class GEOClassifier:
    def fit(self, X_train, y_train):
        """Train on feature matrix and labels"""
        
    def predict(self, X):
        """Return binary predictions"""
        
    def predict_proba(self, X):
        """Return GEO probability scores"""
        
    def predict_entry(self, entry_features):
        """Context-aware prediction for 
           all sources in an entry"""
\end{verbatim}

\subsubsection{ListNet Ranking Model}

\begin{verbatim}
class ListNetRanker(nn.Module):
    def __init__(self, input_dim=389):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x).squeeze(-1)
\end{verbatim}

\subsection{Loss Functions}

\begin{verbatim}
def listnet_loss(scores, true_ranks):
    """ListNet top-1 probability loss"""
    # Convert ranks to relevance (lower rank = 
    # higher relevance)
    relevance = 1.0 / true_ranks
    
    # Softmax over predicted scores
    pred_probs = F.softmax(scores, dim=0)
    
    # Softmax over true relevance
    true_probs = F.softmax(relevance, dim=0)
    
    # Cross-entropy loss
    return -torch.sum(true_probs * 
                      torch.log(pred_probs + 1e-10))

def pairwise_loss(scores, true_ranks, margin=1.0):
    """Margin-based pairwise ranking loss"""
    loss = 0
    n = len(scores)
    for i in range(n):
        for j in range(n):
            if true_ranks[i] < true_ranks[j]:
                # i should be ranked higher than j
                loss += F.relu(margin - 
                              (scores[i] - scores[j]))
    return loss / (n * (n - 1) / 2)
\end{verbatim}

\subsection{Inference Pipeline}

\subsubsection{Single-Source Classification}

\begin{verbatim}
def classify_source(text, model, threshold=0.5):
    # Preprocess
    cleaned = clean_text(text)
    
    # Extract features
    features = extract_features(cleaned)
    
    # Predict
    prob = model.predict_proba(
        features.reshape(1, -1))[0, 1]
    
    return {
        'is_geo': prob > threshold,
        'geo_probability': prob,
        'features': features
    }
\end{verbatim}

\subsubsection{Entry-Level Ranking}

\begin{verbatim}
def rank_sources(sources, model):
    # Extract features for all sources
    features = [extract_features(clean_text(s)) 
                for s in sources]
    X = np.stack(features)
    
    # Get ranking scores
    scores = model.forward(
        torch.tensor(X, dtype=torch.float32))
    
    # Rank by score (higher = more likely GEO)
    rankings = np.argsort(-scores.numpy())
    
    return {
        'rankings': rankings,
        'scores': scores.numpy(),
        'top_geo_idx': rankings[0]
    }
\end{verbatim}

\subsubsection{Threshold Calibration}

\begin{verbatim}
def find_optimal_threshold(y_true, y_scores):
    best_thresh, best_acc = 0, 0
    for thresh in np.linspace(-2, 2, 100):
        preds = (y_scores > thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh
\end{verbatim}

\subsection{Model Serialization}

\begin{verbatim}
# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'threshold': optimal_threshold,
        'pattern_embeddings': pattern_embeddings,
        'config': training_config
    }, f)
\end{verbatim}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

---

## Quick Reference Tables

### Best Model Configurations

| Task | Best Model | Configuration | Accuracy |
|------|------------|---------------|----------|
| Ranking | ListNet | 389-dim features | 87.00% |
| Classification | SVM | PCA 250 | 86.57% |
| Before/After Detection | ListNet | Yuheng-trained | 90.30% |

### Feature Importance (Cohen's d)

| Feature | Cohen's d | Interpretation |
|---------|-----------|----------------|
| Entity Over-Attribution | 0.82 | Large effect |
| Header Stuffing | 0.73 | Medium-large |
| Citation Embedding | 0.61 | Medium |
| Q&A Blocks | 0.57 | Medium |
| Over-Chunking | 0.54 | Medium |

### Category Baseline GEO Scores

| Category | Mean Score |
|----------|------------|
| E-commerce | 0.216 |
| Corporate | 0.209 |
| Affiliate | 0.211 |
| Communication/Social | 0.181 |
| Educational | 0.183 |
| Non-profit | 0.158 |

