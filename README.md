# FashionTrend AI

## Automatic System for Detecting Trends in Fashion Using Machine Learning

FashionTrend AI is a student Skill Lab project that explores how Python,
data analysis, machine learning, and image processing can be used to
identify fashion attributes and analyse how fashion-related interest
changes over time.

The project has two connected analysis pipelines:

1.  **Fashion image analysis** --- process clothing images and extract
    useful information such as image dimensions, visual features,
    clothing category, and dominant colours.
2.  **Fashion trend analysis** --- analyse historical
    popularity/search-interest data, identify emerging or declining
    trends, and experiment with forecasting.

The results will be presented through an interactive **Streamlit
dashboard**.

> **Important:** A collection of fashion product images does not, by
> itself, prove that a style is trending. Trend claims must be based on
> observations over time (for example, weekly Google Trends interest).
> Google Trends values are normalized relative-interest scores, not
> sales, market share, or absolute search counts.

------------------------------------------------------------------------

## Table of contents

1.  [Project goals](#1-project-goals)
2.  [Technology stack](#2-technology-stack)
3.  [Prerequisites](#3-prerequisites)
4.  [Clone the repository](#4-clone-the-repository)
5.  [Set up on macOS](#5-set-up-on-macos)
6.  [Set up on Windows](#6-set-up-on-windows)
7.  [Install project dependencies](#7-install-project-dependencies)
8.  [Recommended project structure](#8-recommended-project-structure)
9.  [Download and organize datasets](#9-download-and-organize-datasets)
10. [Run the project](#10-run-the-project)
11. [Git and team collaboration](#11-git-and-team-collaboration)
12. [Basic terms used in this
    project](#12-basic-terms-used-in-this-project)
13. [Machine learning concepts](#13-machine-learning-concepts)
14. [Evaluation and avoiding data
    leakage](#14-evaluation-and-avoiding-data-leakage)
15. [Troubleshooting](#15-troubleshooting)
16. [Team responsibilities](#16-team-responsibilities)
17. [Development workflow](#17-development-workflow)

------------------------------------------------------------------------

## 1. Project goals

The project aims to:

-   Load and explore fashion product datasets.
-   Clean and prepare structured data using Pandas and NumPy.
-   Process fashion images using Pillow and/or OpenCV.
-   Explore clothing-category classification.
-   Extract dominant image colours using clustering techniques such as
    K-Means.
-   Collect historical fashion-related interest data.
-   Calculate moving averages and percentage changes.
-   Classify trends as emerging, growing, stable, or declining using
    documented rules.
-   Build and evaluate a baseline forecasting model.
-   Display insights in a Streamlit dashboard.

### Intended dashboard features

-   Overview of available fashion categories.
-   Most popular or fastest-growing search terms in the selected
    dataset.
-   Historical trend charts.
-   Trend status and growth indicators.
-   Image upload and image-attribute analysis.
-   Forecast chart, where sufficient historical data is available.

The dashboard must clearly identify whether a result comes from real
collected data, a trained model, or illustrative/demo data.

------------------------------------------------------------------------

## 2. Technology stack

  -----------------------------------------------------------------------
  Tool / library                      What we use it for
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  NumPy                               Numerical arrays and calculations

  Pandas                              Loading, cleaning, filtering,
                                      grouping, and joining tabular data

  Matplotlib                          Creating charts and graphs

  Seaborn                             Statistical data visualizations

  scikit-learn                        Machine-learning algorithms,
                                      preprocessing, and evaluation

  Pillow (PIL)                        Opening, resizing, and saving
                                      images

  OpenCV (`cv2`)                      Image processing and pixel
                                      manipulation

  Streamlit                           Building the interactive web
                                      dashboard

  Jupyter                             Experimenting in notebooks

  Git and GitHub                      Version control and team
                                      collaboration
  -----------------------------------------------------------------------

We will start with lightweight classical ML methods. Install PyTorch or
TensorFlow only if the team later decides to implement a deep-learning
image classifier. Do not install both frameworks by default.

------------------------------------------------------------------------

## 3. Prerequisites

Each team member should have:

-   A laptop running macOS or Windows.
-   A stable internet connection for cloning the repository and
    downloading datasets.
-   Python **3.11 or 3.12** recommended for broad compatibility with
    scientific Python packages.
-   Git.
-   Visual Studio Code.
-   Basic terminal/command-prompt familiarity.

### Check installed tools

Run these commands in a terminal.

``` bash
python --version
git --version
```

On some systems, use:

``` bash
python3 --version
```

If Python is not installed, install Python 3.11 or 3.12 from:

-   Python downloads: https://www.python.org/downloads/
-   Git downloads: https://git-scm.com/downloads
-   VS Code: https://code.visualstudio.com/

During Windows Python installation, enable **Add Python to PATH** if the
installer offers that option.

### VS Code extensions

Open VS Code → Extensions and install:

-   Python (Microsoft)
-   Pylance (Microsoft)
-   Jupyter (Microsoft)

These extensions provide Python execution, code completion, debugging,
and notebook support.

------------------------------------------------------------------------

## 4. Clone the repository

Open Terminal on macOS or PowerShell on Windows.

Replace the URL below with the actual HTTPS or SSH clone URL shown on
the team's GitHub repository page.

``` bash
git clone YOUR_REPOSITORY_URL
cd fashion_trend_AI
```

If the cloned folder has a different name, enter that folder instead.

Open it in VS Code:

``` bash
code .
```

If `code` is not recognized, open VS Code manually and select **File →
Open Folder**.

Check the current Git branch:

``` bash
git branch
```

The repository currently uses these work branches:

-   `data-engineering`
-   `image-analysis`
-   `trend-analysis`
-   `dashboard`

The default branch is `main`.

------------------------------------------------------------------------

## 5. Set up on macOS

The following commands work in the macOS Terminal using zsh or bash.

### Step 1 --- Go to the project folder

``` bash
cd path/to/fashion_trend_AI
```

### Step 2 --- Create a virtual environment

``` bash
python3.11 -m venv .venv
```

If Python 3.11 is not available but Python 3.12 is installed:

``` bash
python3.12 -m venv .venv
```

### Step 3 --- Activate it

``` bash
source .venv/bin/activate
```

When activated, the terminal normally displays `(.venv)` at the
beginning of the prompt.

### Step 4 --- Upgrade pip

``` bash
python -m pip install --upgrade pip
```

### Step 5 --- Install dependencies

Follow [Section 7](#7-install-project-dependencies).

### Step 6 --- Select the correct interpreter in VS Code

1.  Open the project folder in VS Code.
2.  Press `Cmd + Shift + P`.
3.  Search for `Python: Select Interpreter`.
4.  Select the interpreter inside the project's `.venv` folder.
5.  Open a new terminal in VS Code and activate the environment if
    necessary.

### Step 7 --- Verify the environment

``` bash
python --version
python -c "import numpy, pandas, sklearn, streamlit; print('Environment ready')"
```

------------------------------------------------------------------------

## 6. Set up on Windows

The following instructions use **PowerShell**. If using Command Prompt,
see the activation alternative below.

### Step 1 --- Go to the project folder

``` powershell
cd path\to\fashion_trend_AI
```

### Step 2 --- Create a virtual environment

If the Python launcher is installed:

``` powershell
py -3.11 -m venv .venv
```

Or use Python 3.12:

``` powershell
py -3.12 -m venv .venv
```

If `py` is unavailable, try:

``` powershell
python -m venv .venv
```

### Step 3 --- Activate the environment

``` powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation because of the execution policy, use
this temporary setting for the current terminal session:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

``` powershell
.\.venv\Scripts\Activate.ps1
```

This changes the policy only for the current PowerShell process.

**Command Prompt alternative:**

``` bat
.venv\Scripts\activate.bat
```

### Step 4 --- Upgrade pip

``` powershell
python -m pip install --upgrade pip
```

### Step 5 --- Install dependencies

Follow [Section 7](#7-install-project-dependencies).

### Step 6 --- Select the interpreter in VS Code

1.  Open the project folder.
2.  Press `Ctrl + Shift + P`.
3.  Search for `Python: Select Interpreter`.
4.  Select `.venv\Scripts\python.exe`.

### Step 7 --- Verify the environment

``` powershell
python --version
python -c "import numpy, pandas, sklearn, streamlit; print('Environment ready')"
```

------------------------------------------------------------------------

## 7. Install project dependencies

Activate the virtual environment before installing packages.

### Option A --- Install from `requirements.txt` (preferred)

From the repository root:

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This is the standard setup method for every team member.

### Option B --- Install the initial packages individually

Use this only if the team is creating or repairing the dependency file.

``` bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn
python -m pip install pillow opencv-python
python -m pip install streamlit jupyter
```

### What each dependency provides

  Package name in pip   Import name           Project use
  --------------------- --------------------- ----------------------------
  `numpy`               `numpy`               Numerical computation
  `pandas`              `pandas`              Tabular datasets
  `matplotlib`          `matplotlib.pyplot`   Charts
  `seaborn`             `seaborn`             Statistical plots
  `scikit-learn`        `sklearn`             ML algorithms and metrics
  `pillow`              `PIL`                 Image loading and resizing
  `opencv-python`       `cv2`                 Image processing
  `streamlit`           `streamlit`           Dashboard
  `jupyter`             Jupyter tools         Interactive notebooks

### Verify all important packages

``` bash
python -c "import numpy, pandas, matplotlib, seaborn, sklearn, PIL, cv2, streamlit; print('All core packages imported successfully')"
```

If this command finishes without an error, the core environment is
ready.

### Update `requirements.txt`

After installing or changing dependencies:

``` bash
python -m pip freeze > requirements.txt
```

Commit the updated dependency file so other members can reproduce the
environment.

**Team rule:** Do not independently upgrade packages immediately before
a review or demo. Test dependency changes first and coordinate them with
the team.

### Optional deep-learning packages

Only install these when the image-classification module needs them and
the team has agreed on a framework.

PyTorch installation instructions:
https://pytorch.org/get-started/locally/

TensorFlow installation instructions:
https://www.tensorflow.org/install/pip

Check official compatibility guidance for the operating system, Python
version, and hardware before installing. The core project does not
require either framework to begin.

------------------------------------------------------------------------

## 8. Recommended project structure

The repository may evolve as modules are implemented. Use this structure
as the target organization:

``` text
fashion_trend_AI/
│
├── data/
│   ├── raw/
│   │   ├── fashion_images/
│   │   ├── fashion_products.csv
│   │   └── google_trends.csv
│   │
│   └── processed/
│       ├── cleaned_products.csv
│       └── trend_features.csv
│
├── notebooks/
│   ├── day1_numpy.py
│   ├── day1_pandas.py
│   ├── day1_visualization.py
│   ├── 01_data_exploration.ipynb
│   ├── 02_image_analysis.ipynb
│   └── 03_trend_analysis.ipynb
│
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── classification.py
│   ├── trend_detection.py
│   ├── forecasting.py
│   └── evaluation.py
│
├── models/
│   └── README.md
│
├── dashboard/
│   └── app.py
│
├── tests/
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Folder responsibilities

-   `data/raw/`: original downloaded data. Do not manually overwrite
    source data.
-   `data/processed/`: cleaned and transformed data produced by our
    code.
-   `notebooks/`: experiments, explanations, and exploratory analysis.
-   `src/`: reusable Python modules used by the application.
-   `models/`: saved model artifacts, if they are small enough for the
    agreed storage method.
-   `dashboard/`: Streamlit application.
-   `tests/`: automated tests for important functions.
-   `requirements.txt`: Python dependencies.
-   `README.md`: setup and project documentation.

Large datasets and model files should not automatically be committed to
GitHub. Use an agreed shared download location, Git LFS, or another
suitable storage method.

------------------------------------------------------------------------

## 9. Download and organize datasets

We need two different kinds of data: fashion images/product metadata and
popularity observations over time.

### Dataset A --- Fashion product images

A possible starting dataset is the Fashion Product Images dataset:

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

It contains product images and associated product metadata. Review the
dataset page, its license, file structure, and size before downloading.

After downloading and extracting it, organize the files under:

``` text
data/
└── raw/
    ├── fashion_images/
    └── fashion_products.csv
```

The exact CSV filename and image-folder structure may differ. Inspect
the downloaded archive and update the paths in your code accordingly.

### Dataset B --- Historical fashion interest

Use Google Trends:

https://trends.google.com/trends/

Suggested search terms:

-   oversized t-shirt
-   cargo pants
-   wide leg jeans
-   sneakers
-   denim jacket

Suggested collection settings:

-   Geography: India
-   Time range: choose a consistent historical period, ideally 12--24
    months if available.
-   Frequency: weekly, where available.
-   Export: CSV.

Save the exported data under:

``` text
data/raw/google_trends.csv
```

Record the search term, geography, time range, export date, and source
in your project notes. Google Trends normalizes values within the
selected comparison and time window. Do not treat the score as an
absolute number of searches.

### Dataset quality checks

Before model training, inspect:

-   Number of records.
-   Column names and data types.
-   Missing values.
-   Duplicate records.
-   Invalid dates.
-   Category spelling and naming consistency.
-   Whether image files referenced by metadata actually exist.
-   Whether trend observations cover the required time range.

### Keep raw and processed data separate

Never edit the original downloaded dataset directly. Write cleaned
output into `data/processed/` so the process can be repeated.

------------------------------------------------------------------------

## 10. Run the project

### Run a Python script

Activate the virtual environment, then run a script from the repository
root:

``` bash
python notebooks/day1_numpy.py
python notebooks/day1_pandas.py
python notebooks/day1_visualization.py
```

### Run Jupyter

``` bash
jupyter notebook
```

Or:

``` bash
jupyter lab
```

Open the relevant notebook and select the Python kernel from the
project's `.venv`.

### Run the Streamlit dashboard

From the repository root:

``` bash
streamlit run dashboard/app.py
```

Streamlit normally opens the dashboard in your browser. If it does not,
use the local URL printed in the terminal, usually:

``` text
http://localhost:8501
```

To stop the dashboard, focus the terminal and press `Ctrl + C`.

### Run a script from VS Code

1.  Open the `.py` file.
2.  Confirm that the selected interpreter is `.venv`.
3.  Click **Run Python File** or use the integrated terminal.
4.  Read the output and any error messages.

------------------------------------------------------------------------

## 11. Git and team collaboration

The shared repository uses `main` as the default branch and separate
work branches:

-   `data-engineering`
-   `image-analysis`
-   `trend-analysis`
-   `dashboard`

### First-time Git identity setup

Use your own name and GitHub email:

``` bash
git config --global user.name "Your Name"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

### Start work on your assigned branch

``` bash
git switch your-branch-name
```

For example:

``` bash
git switch trend-analysis
```

### Before starting new work, get the latest branch changes

``` bash
git pull origin your-branch-name
```

### Save and push your work

``` bash
git status
git add path/to/your/files
git commit -m "Describe the change"
git push origin your-branch-name
```

Then open a Pull Request on GitHub to merge the work into `main`,
following the team's review process.

### Important collaboration rules

1.  Do not commit directly to `main` unless the team explicitly agrees.
2.  Pull the latest changes before starting work.
3.  Commit small, meaningful changes.
4.  Do not commit `.venv/`, secrets, cache folders, or unnecessary
    generated files.
5.  Coordinate changes to `requirements.txt`.
6.  Do not have two members independently edit the same file without
    coordination.
7.  Review another member's code before merging.
8.  Test the integrated application after each merge.

### Recommended `.gitignore`

Create `.gitignore` in the repository root:

``` gitignore
# Python environments
.venv/
venv/
env/

# Python cache and compiled files
__pycache__/
*.py[cod]
.pytest_cache/

# Jupyter checkpoints
.ipynb_checkpoints/

# Environment variables and secrets
.env
.env.*
!.env.example

# Operating system files
.DS_Store
Thumbs.db

# Editor settings (team may choose to share selected settings)
.vscode/

# Logs
*.log

# Generated model files
*.pkl
*.joblib
*.h5
*.keras
*.pt
*.pth

# Large or downloaded datasets
data/raw/*
!data/raw/.gitkeep

# Generated processed data (commit selectively if useful)
data/processed/*
!data/processed/.gitkeep
```

If the team decides to version a small model or processed sample
dataset, explicitly add that file to Git and document why. Do not
force-add private data or secrets.

If the application requires credentials, store them in `.env` locally
and provide an `.env.example` containing placeholder variable names
only. Never commit real credentials.

------------------------------------------------------------------------

## 12. Basic terms used in this project

This glossary explains the important terms in beginner-friendly language
and shows where each one is used.

  -----------------------------------------------------------------------
  Term                    Meaning in simple words How we use it
  ----------------------- ----------------------- -----------------------
  Dataset                 An organized collection Fashion images, product
                          of information          CSV, trend CSV

  Data point / record     One row or one          One product or one
                          observation             weekly trend
                                                  observation

  Feature                 Information given to a  Image pixels, colour,
                          model                   category, previous
                                                  interest values

  Label / target          The answer a supervised Clothing category or
                          model learns to predict future interest

  Attribute               A property describing   Brand, colour,
                          an item                 category, season

  Data preprocessing      Preparing raw data for  Cleaning CSV data and
                          analysis                resizing images

  Missing value           A field with no         Product with no colour
                          recorded value          value

  Duplicate               A repeated record       Same product row
                                                  appearing twice

  DataFrame               A table-like Pandas     Product and trend
                          structure               tables

  NumPy array             A numerical array       Pixel values and model
                          structure               inputs

  Pixel                   The smallest            Used to analyse
                          addressable unit of a   clothing images
                          digital image           

  RGB                     Red, Green, Blue colour Representing image
                          channels                colours

  Image resizing          Changing image          Making input sizes
                          dimensions              consistent

  Normalization           Scaling values to a     Preparing numerical
                          common range            inputs for models

  Feature extraction      Turning raw data into   Extracting dominant
                          useful measurable       colours from an image
                          information             

  Algorithm               A method for solving a  Random Forest, K-Means,
                          problem                 Linear Regression

  Machine Learning model  A learned mathematical  Clothing classifier or
                          pattern used for        forecasting model
                          prediction              

  Training                Learning patterns from  Fitting the classifier
                          training data           or regressor

  Testing                 Checking performance on Measuring performance
                          held-out data           on unseen observations

  Prediction / inference  Using a trained model   Predicting a clothing
                          on new input            category

  Classification          Predicting a category   T-shirt, jeans, shoes
                          or class                

  Regression              Predicting a numerical  Future search-interest
                          value                   score

  Clustering              Grouping similar        Grouping similar
                          observations without    colours or trend
                          given labels            patterns

  K-Means                 An algorithm that       Dominant colour
                          groups data into K      extraction and optional
                          clusters                trend grouping

  CNN                     Convolutional Neural    Optional advanced
                          Network, a              clothing-image
                          deep-learning model for classification
                          image patterns          

  Moving average          Average of values over  Smoothing weekly trend
                          a moving time window    fluctuations

  Time series             Data recorded in time   Weekly Google Trends
                          order                   observations

  Trend                   A sustained direction   Increasing, stable, or
                          or pattern over time    decreasing interest

  Forecasting             Estimating future       Predicting future
                          values from past        search interest
                          observations            

  Model evaluation        Measuring how well a    Accuracy, F1-score,
                          model works             MAE, RMSE

  Dashboard               An interactive screen   Streamlit application
                          showing data and        
                          results                 

  API                     A defined way for       Optional future
                          software components to  connection to data
                          communicate             services

  CSV                     Comma-Separated Values  Storing product and
                          file                    trend tables

  Virtual environment     An isolated Python      Keeping team
                          package environment     dependencies separate

  Dependency              An external package     NumPy, Pandas,
                          required by the project Streamlit

  Git                     Version-control         Tracking source-code
                          software                changes

  Branch                  An independent line of  Each member's assigned
                          code development        work

  Commit                  A recorded set of code  Saving progress in Git
                          changes                 

  Merge / Pull Request    Combining reviewed code Integrating member
                          changes                 modules
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 13. Machine learning concepts

### 13.1 Supervised learning

A model learns from examples that already contain the correct answer
(label).

**Example in this project:** Give a model fashion images with known
category labels. It learns to predict the category of a new image.

-   Input (`X`): image-derived features or image pixels.
-   Target (`y`): known clothing category.

### 13.2 Unsupervised learning

A model looks for structure in data without being given the correct
category for every observation.

**Example in this project:** K-Means groups similar image colours into
clusters. The team can interpret the cluster centres as representative
colours.

### 13.3 Classification

Classification predicts a class or category.

Example:

``` text
Input image → Model → "T-shirt"
```

Useful metrics:

-   Accuracy: proportion of predictions that are correct.
-   Precision: among items predicted as a class, how many are correct?
-   Recall: among all actual items in a class, how many did the model
    find?
-   F1-score: a combined measure of precision and recall.

### 13.4 Regression

Regression predicts a numerical value.

Example:

``` text
Previous weekly interest values → Model → Predicted next-week interest
```

Useful metrics:

-   MAE: average absolute prediction error.
-   RMSE: square root of the average squared prediction error.

Lower MAE and RMSE generally indicate smaller prediction errors when
models are evaluated on the same target scale and test period.

### 13.5 Clustering

Clustering groups similar data points.

Example:

-   Convert image pixels into colour values.
-   Run K-Means with a selected number of clusters.
-   Interpret cluster centres as dominant colours.

### 13.6 Moving average

A moving average smooths short-term fluctuations.

For a 3-week moving average:

``` text
Moving average = (Week 1 + Week 2 + Week 3) / 3
```

As new observations arrive, the window moves forward.

### 13.7 Percentage change

Percentage change compares a new value with an earlier value:

``` text
Percentage change = ((New value - Old value) / Old value) * 100
```

For example, a change from 20 to 30 is a 50% increase.

If the old value is zero, this formula is undefined. Handle zero values
explicitly in code.

### 13.8 Emerging, growing, stable, and declining

These are trend-status labels that our project defines using transparent
rules based on historical observations.

For example, the team may define thresholds using smoothed growth over a
chosen period. These thresholds are project rules, not universal
fashion-industry definitions.

A trend should not be called emerging solely because its last
observation is higher than its first observation. Consider the full time
series, the selected comparison window, and the amount of data
available.

------------------------------------------------------------------------

## 14. Evaluation and avoiding data leakage

Evaluation is essential. A model that runs successfully is not
automatically a good model.

### Image classification

-   Split labelled image data into training, validation, and test sets.
-   Keep related or duplicate images from leaking across splits where
    possible.
-   Report accuracy, precision, recall, and F1-score.
-   Inspect a confusion matrix to understand which categories are
    confused.

### Trend forecasting

-   Keep observations in chronological order.
-   Train on earlier observations and test on later observations.
-   Compare the model with a simple baseline, such as predicting that
    the next value equals the latest observed value.
-   Report MAE and RMSE.
-   Do not randomly shuffle time-series rows before splitting.

### Data leakage

Data leakage happens when information from the test period accidentally
influences model training or preprocessing.

Examples:

-   Calculating preprocessing statistics using the full dataset before
    splitting.
-   Using future popularity values to create features for a past
    prediction.
-   Randomly splitting overlapping time windows so that nearly identical
    observations appear in both training and testing sets.

Fit preprocessing steps on training data only, then apply the fitted
transformations to validation and test data.

### Responsible interpretation

-   Google Trends scores are relative, not absolute demand.
-   A rise in search interest does not prove a rise in sales.
-   A fashion image dataset may be biased toward its source, categories,
    geography, or collection period.
-   Forecasts are estimates, not guarantees.
-   Clearly state dataset sources, collection periods, limitations, and
    evaluation results in the final presentation.

------------------------------------------------------------------------

## 15. Troubleshooting

### Problem: `python` or `python3` is not recognized

**Windows:**

``` powershell
py --version
```

If this works, use `py -3.11` to create the virtual environment.
Otherwise install Python and ensure it is available on PATH.

**macOS:**

``` bash
python3 --version
```

Install a supported Python version if needed.

### Problem: `ModuleNotFoundError`

The package may not be installed in the active environment, or VS Code
may be using a different interpreter.

1.  Activate `.venv`.
2.  Select `.venv` in VS Code's Python interpreter selector.
3.  Run:

``` bash
python -m pip install -r requirements.txt
```

4.  Verify the import again.

### Problem: Virtual environment activation is blocked on Windows

In PowerShell, run:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively use Command Prompt:

``` bat
.venv\Scripts\activate.bat
```

### Problem: Matplotlib graph window does not appear

-   Ensure the script reaches `plt.show()`.
-   Close any existing graph window.
-   Run the script again.
-   If running without a graphical display, save the graph instead:

``` python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [10, 20, 15])
plt.savefig("trend_chart.png", dpi=150, bbox_inches="tight")
```

### Problem: Script appears to hang while displaying a graph

Some Matplotlib backends wait until the plot window is closed. Close the
graph window to allow execution to continue. If needed, use
`plt.savefig()` instead of interactive display.

### Problem: Streamlit command is not found

Run Streamlit through the active Python interpreter:

``` bash
python -m streamlit run dashboard/app.py
```

### Problem: Git says there are merge conflicts

1.  Stop and read the conflict messages.
2.  Open the files marked as conflicted.
3.  Resolve the conflict with the other contributor if necessary.
4.  Run the relevant tests.
5.  Stage and commit the resolved files.

Do not blindly choose one side of a conflict without understanding which
changes would be lost.

### Problem: Installation fails on a new Python version

Use the team's agreed Python 3.11 or 3.12 environment, recreate `.venv`,
and reinstall from `requirements.txt`. Check package compatibility
before changing Python versions.

------------------------------------------------------------------------

## 16. Team responsibilities

  -----------------------------------------------------------------------
  Member                  Ownership               Main deliverables
  ----------------------- ----------------------- -----------------------
  Member 1 --- Data       Dataset collection and  Clean data,
  Engineering             preparation             preprocessing module,
                                                  data-quality report

  Member 2 --- Image      Image processing and    Image feature
  Analysis                classification          extraction,
                                                  classification module,
                                                  model evaluation

  Member 3 --- Trend      Historical interest and Trend detection,
  Analytics               forecasting             forecasting, trend
                                                  evaluation

  Member 4 ---            Dashboard and           Streamlit app,
  Application Integration integration             visualizations,
                                                  integration tests
  -----------------------------------------------------------------------

All members should understand the overall architecture and review each
other's work. Module ownership does not mean that only one member should
understand that module.

------------------------------------------------------------------------

## 17. Development workflow

For every feature, follow this cycle:

1.  Understand the problem and expected output.
2.  Inspect the data before writing the algorithm.
3.  Implement a small baseline.
4.  Test the implementation with known examples.
5.  Evaluate the result with suitable metrics.
6.  Document assumptions and limitations.
7.  Commit the code to the assigned Git branch.
8.  Request review and integrate through a Pull Request.

### Suggested milestones

  -----------------------------------------------------------------------
  Milestone                           Expected outcome
  ----------------------------------- -----------------------------------
  Week 1                              Environment, datasets, data
                                      exploration, initial dashboard
                                      skeleton

  Week 2                              Image processing, classification
                                      baseline, trend detection

  Week 3                              Forecasting, dashboard integration,
                                      end-to-end prototype

  Week 4                              Testing, evaluation, documentation,
                                      and presentation
  -----------------------------------------------------------------------

For a three-week schedule, prioritize the working baseline and reduce
optional deep-learning enhancements.

------------------------------------------------------------------------

## Final setup checklist

Before starting development, each team member should confirm:

-   [ ] Git installed and repository cloned.
-   [ ] Python 3.11 or 3.12 installed.
-   [ ] Project virtual environment created and activated.
-   [ ] VS Code Python and Jupyter extensions installed.
-   [ ] `requirements.txt` dependencies installed successfully.
-   [ ] Core package import test passes.
-   [ ] Correct Git branch selected.
-   [ ] Dataset files placed in the agreed folders.
-   [ ] No secrets or virtual-environment files committed.
-   [ ] A sample Python script and Streamlit app can run.

**The project is ready for development when every team member can
reproduce the environment from this README and run the starter code
successfully.**

------------------------------------------------------------------------

## References

-   Python: https://www.python.org/
-   NumPy: https://numpy.org/doc/
-   Pandas: https://pandas.pydata.org/docs/
-   Matplotlib: https://matplotlib.org/stable/
-   Seaborn: https://seaborn.pydata.org/
-   scikit-learn: https://scikit-learn.org/stable/
-   Pillow: https://pillow.readthedocs.io/
-   OpenCV: https://docs.opencv.org/
-   Streamlit: https://docs.streamlit.io/
-   Jupyter: https://docs.jupyter.org/
-   Git: https://git-scm.com/doc
-   GitHub: https://docs.github.com/
-   Google Trends: https://trends.google.com/trends/
