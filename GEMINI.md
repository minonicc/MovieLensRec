# Gemini Code Guide: MovieLens Recommendation Project

## Project Overview

This project aims to build a movie recommendation system using the MovieLens-32M dataset. The current focus is on the initial data analysis and exploration phase, which serves as the foundation for developing recommendation algorithms (recall models).

The project is structured to support a typical machine learning workflow, from data analysis to model implementation.

- **Technology Stack:** Python, Pandas, Matplotlib, Scikit-learn, Jupyter, Conda
- **Dataset:** MovieLens 32M, containing movie ratings, titles, genres, and tags.

## Guidelines

- **Language:** Please respond in Chinese.
- **Code Comments:** All code comments must be written in Chinese.
- **Preserve Comments:** 修改代码逻辑时，应同步更新相关注释以保持一致；但在代码逻辑未变动的情况下，严禁随意修改、重写、简化或删除既有注释。在进行修复或功能改进时，应尽力保留与变动无关的原始注释及其格式。

## Project Structure

- `data/ml-32m/`: Contains the raw MovieLens dataset files (`movies.csv`, `ratings.csv`, etc.).
- `notebooks/`: Houses Jupyter notebooks for data analysis, experimentation, and visualization. The primary analysis starts in `01_data_analysis.ipynb`.
- `src/`: Intended for storing modular Python source code for the recommendation models, data processing pipelines, and other utilities.
- `__pycache__/`: Python's directory for storing bytecode cache (can be ignored).

## Environment Setup

The project uses a Conda environment to manage dependencies.

1.  **Environment Name:** `movielens-rec`
2.  **Python Version:** `3.9`
3.  **Key Libraries:**
    - `pandas` & `numpy` for data manipulation.
    - `matplotlib` & `seaborn` for visualization.
    - `scikit-learn` for machine learning utilities.
    - `jupyter` for interactive analysis.

To set up the environment from scratch, configure conda to use the Tsinghua mirror and then create the environment:

```bash
# Configure Conda to use Tsinghua mirrors (optional, for faster downloads in China)
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

# Create the conda environment with all necessary packages
conda create --name movielens-rec -y python=3.9 pandas numpy matplotlib scikit-learn jupyter
```

## Running the Analysis

To begin the data analysis, you need to activate the conda environment and start the Jupyter Notebook server.

1.  **Activate the Conda environment:**
    ```bash
    conda activate movielens-rec
    ```

2.  **Start the Jupyter server:**
    ```bash
    jupyter notebook
    ```
    This will open a new tab in your web browser.

3.  **Open the notebook:**
    - Navigate to the `notebooks` directory in the Jupyter interface.
    - Click on `01_data_analysis.ipynb` to open it.
    - If using VSCode, open the notebook file and select the `movielens-rec` conda environment as the kernel when prompted.
