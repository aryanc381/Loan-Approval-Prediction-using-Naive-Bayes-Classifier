# Loan Eligibility Prediction using Naive Bayes Classifier

This project aims to predict the loan eligibility of individuals using a Naive Bayes Classifier (NBC). The model is built and evaluated using various features of the dataset to determine the eligibility of loan applicants.

## Project Overview

Loan approval prediction is a critical task in the financial sector. By leveraging machine learning techniques, we can automate and enhance the decision-making process, reducing manual errors and increasing efficiency. This project utilizes the Naive Bayes Classifier, a simple yet effective probabilistic classifier, to predict whether a loan applicant is eligible for a loan.

## Features

- **Preprocessing:** Data cleaning and preprocessing to handle missing values, outliers, and categorical variables.
- **Feature Selection:** Selecting relevant features that contribute significantly to the prediction model.
- **Model Building:** Implementing the Naive Bayes Classifier to predict loan eligibility.
- **Model Evaluation:** Evaluating the model's performance using metrics like accuracy, precision, recall, and F1-score.

## Dataset

The dataset used in this project contains various features of loan applicants, including:

- Applicant's income
- Co-applicant's income
- Loan amount
- Loan term
- Credit history
- Property area
- Gender
- Marital status
- Education
- Employment status

## Requirements

To run this project, you need the following Python libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/LoanApprovalPrediction
cd LoanApprovalPrediction
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:

```bash
jupyter notebook main.ipynb
```

4. Follow the steps in the notebook to preprocess the data, build and evaluate the Naive Bayes model.

## Results

The model's performance is evaluated using various metrics, providing insights into its accuracy and reliability in predicting loan eligibility. Detailed results and visualizations are included in the Jupyter Notebook.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please open an issue or create a pull request.

## License

This project is licensed under the MIT License.

---

Feel free to customize this README file further based on specific details and preferences for your project.
