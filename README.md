# A Custom Polynomial Regression Model for Powerlifting Performance Evaluation: 

## Improving upon the Wilks Coefficient

Abstract: This study proposes a novel approach to evaluate powerlifting performance using a custom polynomial regression model. The proposed method is an improvement over the widely-used Wilks Coefficient, as it is capable of capturing non-linear relationships between body weight and total weight lifted in a more flexible and accurate manner. Additionally, our model is agnostic to gender, making it applicable to a broader range of athletes. The results show that the proposed custom polynomial regression model outperforms the Wilks Coefficient method in terms of mean squared error and R2 score, demonstrating its potential as a more effective powerlifting performance metric.

1. Introduction The Wilks Coefficient is a widely-used performance metric in the powerlifting community, which aims to enable fair comparisons of strength across different weight classes and genders. However, the Wilks Coefficient has some limitations, such as its reliance on gender-specific coefficients and an inability to effectively model non-linear relationships between body weight and total weight lifted. In this study, we propose a custom polynomial regression model to address these limitations and offer an improved method for evaluating powerlifting performance.

2. Methodology Our approach involved the following steps:	

   2.1. Data Collection and Preprocessing We used a publicly available dataset containing powerlifting competition results. This dataset included information about athletes' body weight and total weight lifted (squat, bench press, and deadlift combined). We converted the body weight and total weight columns to numeric types and dropped any rows with missing values.

   2.2. Model Development We employed a polynomial regression model with a degree of 3, as it provides a balance between model complexity and the risk of overfitting. The dataset was split into training (80%) and testing (20%) sets, and the model was fitted using the training data.

   2.3. Model Evaluation We evaluated the model's performance using mean squared error and R2 score for both training and testing sets. These metrics helped assess the model's ability to generalize and its overall accuracy in predicting the relationship between body weight and total weight lifted.

3. Results: The custom polynomial regression model demonstrated improved performance compared to the Wilks Coefficient method. Our model had lower mean squared error and higher R2 scores for both training and testing sets. This indicates that the custom polynomial regression model can more accurately capture the relationship between body weight and total weight lifted.

$$
\text{Total Lifted} = \beta_0 + \beta_1 \cdot (\text{Bodyweight}) + \beta_2 \cdot (\text{Bodyweight})^2 + \beta_3 \cdot (\text{Bodyweight})^3 \\
\text{Custom Points} = \frac{\text{Actual Total Lifted}}{\text{Predicted Total Lifted}} \cdot 500
$$

4. Discussion The proposed custom polynomial regression model offers several advantages over the Wilks Coefficient method:

   - Our model is agnostic to gender, making it more inclusive and applicable to a broader range of athletes.

   - The custom polynomial regression model can effectively capture non-linear relationships between body weight and total weight lifted, providing a more accurate representation of an athlete's performance.

   - The model's improved mean squared error and R2 scores demonstrate its potential as a more effective powerlifting performance metric.

This page uses data from the OpenPowerlifting project, https://www.openpowerlifting.org. You may download a copy of the data at https://data.openpowerlifting.org.

## Instructions

To run the script, you'll need Python installed on your system and some Python packages. Here are step-by-step instructions on how to install the required packages and run the script:

1. Install Python: If you don't have Python installed, you can download and install it from the official website: https://www.python.org/downloads/. The script requires Python 3.6 or newer.

2. Install required packages: Open a terminal or command prompt, and run the following command to install the necessary packages:

```bash
pip install pandas numpy scikit-learn matplotlib
```

This will install the following packages:

- pandas: A library for data manipulation and analysis.
- numpy: A library for numerical computing in Python.
- scikit-learn: A library for machine learning and data mining in Python.
- matplotlib: A library for creating static, animated, and interactive visualizations in Python.

3. Save the script: Copy the provided script into a new file, and save it with a `.py` extension, e.g., `polylift_performance.py`.

4. Run the script: In the terminal or command prompt, navigate to the folder where you saved the script, and run the following command:

```bash
python polylift_performance.py
```

5. Follow the prompts: The script will ask you to enter your gender, body weight, and total lifted weight. It will then output your custom points based on the trained polynomial regression models.

Remember to download the dataset from here https://data.openpowerlifting.org and place the CSV file into the directory. Make sure to rename it openpowerlifting.csv
