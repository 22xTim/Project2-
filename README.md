# CIS 3400 – Advanced Programming
## Project 2: Python Application Development

**Student Name:** [Timothy White]  
**Project Title:** SleepWise – Sleep Quality Predictor  
**Course:** CIS 3400 – Advanced Programming  
**Final Presentation:** December 8, 2026

---

## 1. Project Overview

**Application Name:** SleepWise

**Project Option:** Predictive ML

### Problem Statement
Many people know that habits like caffeine, alcohol, exercise, and bedtime affect how well they sleep, but it is hard to tell which habits matter most for them or what would actually improve their sleep. SleepWise uses a machine learning model trained on sleep and lifestyle data to predict a person's sleep quality from their daily habits. Instead of stopping at a prediction, the application shows which factors drove the result and lets the user test "what if" changes, such as cutting evening caffeine or going to bed earlier. This turns a prediction into practical, actionable guidance. Better sleep supports health, focus, and academic or work performance, which makes the problem worth solving.

### Intended Users
College students and working adults who want to understand how their habits affect sleep. Users enter information about their typical habits, receive a predicted sleep quality rating, see which habits influence the prediction most, and explore which changes would give the biggest improvement.

### Unique Features
- Goes beyond a single prediction by explaining the factors behind each result.
- Includes a "what if" panel that re-runs the model with adjusted habits and shows the predicted change.
- Ranks habit changes by their estimated impact so users know where to start.
- Built as an interactive web app with a trained and saved model, not a one-off notebook demonstration.

## 2. Application Features

- **Feature 1 – Habit input form:** Users enter bedtime, wake time, caffeine intake, alcohol intake, exercise frequency, and similar habits using sliders and dropdowns.
- **Feature 2 – Sleep quality prediction:** The trained model returns a predicted sleep quality category (Good / Fair / Poor) with a probability or score.
- **Feature 3 – Factor explanation:** A chart shows which inputs contributed most to the prediction.
- **Feature 4 –** Users change one or more habits and see the updated prediction, along with a ranked list of the changes with the largest predicted benefit.

Users interact with the application through a Streamlit page. They enter their habits, click to generate a prediction, and then review the predicted result, the explanation chart, and the what-if recommendations.

## 3. Technologies Used

| Technology | Purpose |
| Python | Main programming language |
| Streamlit | Web interface and application framework |
| Pandas / NumPy | Data loading, cleaning, and feature preparation |
| scikit-learn | Training, evaluating, and using the prediction model |
| joblib | Saving and loading the trained model |
| Matplotlib | Charts for factor importance and what-if results |
| Jupyter Notebook | Data exploration and model training |

### How the Application Works

1. The user enters or selects their habits (bedtime, caffeine, alcohol, exercise, and so on).
2. The Streamlit interface passes the inputs to Python.
3. Python functions clean and format the inputs to match the model's training data.
4. The saved scikit-learn model predicts sleep quality and its probability.
5. The application displays the prediction, a factor-importance chart, and ranked recommendations.
6. When the user changes a habit in the what-if panel, steps 3–5 repeat with the new values to show the predicted change.
