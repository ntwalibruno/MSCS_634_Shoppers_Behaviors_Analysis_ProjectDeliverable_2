**Author**: Ntwali Bruno Bahongere  
**Course**: Advanced Big Data and Data Mining  
**Project**: Residency Project - Shoppers Behavior Analysis
**Date**: July 2025

## Project Overview

This project conducts a comprehensive analysis of online shopping behavior to understand customer patterns, predict revenue generation, and build production-ready machine learning models. Using a dataset of e-commerce website sessions, we perform data cleaning, exploratory data analysis (EDA), advanced feature engineering, and deploy both classification and regression models with rigorous evaluation.

## Dataset Summary

**Source**: [Shoppers Behavior and Revenue Dataset](https://www.kaggle.com/datasets/subhajournal/shoppers-behavior-and-revenue) from Kaggle

**Dataset Characteristics**:
- **Original Size**: 12,330 sessions × 18 features
- **Final Cleaned Size**: 12,205 sessions × 18 features (99.0% data retention)
- **Post Feature Engineering**: 12,205 sessions × 50+ features
- **Target Variables**: 
  - `Revenue` (Boolean - Classification target)
  - `PageValues` (Numeric - Regression target)
- **Time Period**: 10 months of e-commerce data
- **Data Quality**: Excellent (no missing values, minimal duplicates)

### Original Feature Categories:
- **Numerical Features (14)**: Page visit counts, session durations, bounce/exit rates, page values
- **Categorical Features (4)**: Month, visitor type, weekend indicator, revenue outcome

## Key Insights Discovered

### 1. Revenue Conversion Patterns
- **Overall Conversion Rate**: 15.6% of sessions result in purchases
- **Class Imbalance**: 5.4:1 ratio (non-revenue to revenue sessions)
- **Weekend Effect**: Weekend sessions show better conversion patterns
- **Seasonal Trends**: Strong monthly variation in conversion rates

### 2. Customer Behavior Segmentation
- **New vs Returning Visitors**: New visitors demonstrate higher conversion potential
- **High-Value Session Characteristics**:
  - Significantly longer total session duration
  - Higher product page engagement
  - Lower bounce and exit rates
  - Elevated page value metrics

### 3. Feature Relationships & Correlations
- **Strong Negative Predictors**: ExitRates (-0.237), BounceRates (-0.176)
- **Positive Predictors**: ProductRelated_Duration (+0.188), PageValues
- **High Feature Correlations**: Page counts strongly correlated with durations (0.7-0.8)
- **Zero-Inflation Patterns**: Duration features show significant zero-value presence

### 4. Model Performance Results
- **Classification Model (Revenue Prediction)**:
  - ROC-AUC: 0.90+ (Excellent performance)
  - Accuracy: 85%+ with strong generalization
  
- **Regression Model (PageValues Prediction)**:
  - R²: 0.999+ (Near-perfect prediction)
  - RMSE: <1.0 (Very low prediction error)
  
## Data Cleaning and Preprocessing Methodology

### Phase 1: Data Quality Assessment
1. **Missing Values Analysis**: 
   - Comprehensive check across all 18 features
   - No missing values detected in the original dataset
   
2. **Duplicate Detection**: 
   - Identified 125 duplicate records (1.0% of data)
   - Systematic analysis of duplicate patterns
   
3. **Data Type Validation**: 
   - Verified appropriate data types for each feature
   - Ensured numerical and categorical features are properly formatted
   
4. **Outlier Detection**: 
   - Applied IQR method (3×IQR threshold) for outlier identification
   - Statistical analysis across all numerical features

### Phase 2: Data Cleaning Implementation
1. **Missing Values Handling**:
   - No missing values required imputation
   - Robust framework implemented for future datasets
   
2. **Duplicate Removal**:
   - Removed 125 duplicate rows using pandas drop_duplicates()
   - Preserved 99.0% of original data integrity
   
3. **Data Validation**:
   - Verified cleaned dataset structure and quality
   - Confirmed data types and statistical properties

### Phase 3: Advanced Data Preparation
- Created cleaned dataset (`df_cleaned`) for analysis
- Configured visualization environment with matplotlib and seaborn
- Established consistent plotting styles and parameters
- Prepared data structures for feature engineering pipeline

## Exploratory Data Analysis (EDA)

### 1. Numerical Features Analysis
- **Distribution Visualization**: 3×3 grid of histograms with KDE overlays
- **Statistical Summary**: Detailed statistics including mean, median, skewness, kurtosis
- **Skewness Assessment**: Identified heavily right-skewed distributions requiring transformation
- **Zero-Inflation Detection**: Systematic analysis of features with >10% zero values
- **Statistical Interpretation**: Automated interpretation of distribution shapes

### 2. Categorical Features Analysis  
- **Distribution Analysis**: 2×3 grid visualizing all categorical features
- **High-Cardinality Handling**: Top 10 focus for OperatingSystems, Browser, Region
- **Value Frequency**: Detailed breakdown with percentages for all unique values
- **Comprehensive Statistics**: Complete analysis of categorical variable distributions

### 3. Advanced Correlation Analysis
- **Correlation Matrix**: Masked heatmap showing only lower triangle for clarity
- **High Correlation Detection**: Automated identification of correlations >0.7
- **Target Variable Relationships**: Comprehensive analysis of feature-revenue correlations
- **Statistical Significance Testing**: T-tests with p-value annotations for key relationships

### 4. Advanced Visualizations & Statistical Testing
- Distribution plots with statistical overlays (mean, median, KDE)
- Categorical bar charts with frequency analysis
- Correlation heatmap with professional formatting
- Box plots showing feature distributions by revenue outcome
- Statistical significance testing with automated interpretation
- Professional color schemes and consistent formatting

## Advanced Feature Engineering

### 1. Page Engagement Metrics
- **Total_Pages**: Sum of all page visit types
- **Total_Duration**: Aggregate session duration
- **Avg_Time_Per_Page**: Average engagement per page
- **PageValue_Per_Duration**: Value efficiency metrics
- **PageValue_Per_Page**: Value per page interaction

### 2. User Behavior Patterns
- **High_Bounce/High_Exit**: Binary indicators based on median thresholds
- **Bounce_Exit_Score**: Combined bounce-exit behavior metric
- **Product_Focus_Ratio**: Ratio of product-related engagement
- **Session_Depth**: Weighted page depth calculation

### 3. Temporal and Seasonal Features
- **Month_Numeric**: Numerical month encoding
- **Season**: Categorical seasonal grouping (Winter, Spring, Summer, Fall)
- **Holiday_Season**: Binary indicator for November-December
- **Special_Period**: Combined special day and holiday indicator

### 4. Technology and User Profile Features
- **Popular_Browser/Popular_OS**: Binary indicators based on frequency
- **Is_Returning/Is_New**: Visitor type binary encoding
- **Weekend_Shopping**: Weekend indicator

### 5. Interaction and Composite Features
- **High_Value_Session**: Multi-criteria high-value session identifier
- **Engagement_Score**: Weighted composite engagement metric
- **High_Risk_Exit**: Risk indicator based on bounce/exit percentiles
- **Page_Diversity**: Count of different page types visited

### 6. Statistical Transformations
- **Log Transformations**: Applied to skewed duration features using log1p
- **Standardization**: StandardScaler applied to key continuous features
- **Feature Scaling**: Normalized versions of engagement metrics

**Total Engineered Features**: 32 new features created, expanding dataset to 50+ total features

## Machine Learning Models & Evaluation

### 1. Model Architecture
- **Classification Model**: Logistic Regression for Revenue prediction
- **Regression Model**: Lasso Regression for PageValues prediction
- **Data Preparation**: Stratified train-test splits with proper scaling
- **Feature Selection**: Systematic exclusion of redundant and target variables

### 2. Model Performance Metrics

#### Classification Model (Revenue Prediction)
- **Primary Metric**: ROC-AUC Score > 0.90 (Excellent)
- **Accuracy**: 85%+ with balanced precision-recall
- **Cross-Validation**: 5-fold stratified CV with consistent performance
- **Stability**: Very stable across all CV folds
- **Generalization**: Excellent - minimal overfitting

#### Regression Model (PageValues Prediction)
- **Primary Metric**: R² > 0.999 (Near-perfect prediction)
- **RMSE**: <1.0 (Very low prediction error)
- **Cross-Validation**: 5-fold CV with exceptional consistency
- **Stability**: Extremely stable performance
- **Generalization**: Excellent - no overfitting detected

### 3. Advanced Model Evaluation

#### Cross-Validation Analysis
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Overfitting Assessment**: Train vs test performance comparison
- **Stability Analysis**: Coefficient of variation calculations
- **Visualization**: Professional box plots of CV results

#### Performance Visualizations
- **ROC Curves**: Classification performance with AUC visualization
- **Confusion Matrix**: Heatmap showing classification accuracy
- **Actual vs Predicted**: Scatter plots for regression performance
- **Residuals Analysis**: Error distribution assessment

## Challenges Encountered and Solutions

### Challenge 1: Class Imbalance in Revenue Prediction
**Problem**: Highly imbalanced target variable (5.4:1 ratio)
**Solution**: 
- Used stratified sampling in train-test splits
- Applied appropriate evaluation metrics (ROC-AUC, precision-recall)
- Recommended class weights and SMOTE for future improvements

### Challenge 2: Highly Skewed Feature Distributions
**Problem**: Most numerical features heavily right-skewed
**Solution**:
- Applied log1p transformations to skewed features
- Used robust statistical measures (median, IQR)
- Implemented appropriate visualization techniques

### Challenge 3: Zero-Inflated Features
**Problem**: Many duration features contain significant zero values
**Solution**:
- Systematic zero-inflation analysis (>10% threshold)
- Special handling in feature engineering (adding 1 to denominators)
- Flagged for special treatment in modeling

### Challenge 4: High-Cardinality Categorical Features
**Problem**: Features like OperatingSystems, Browser, Region have many unique values
**Solution**:
- Created popularity-based binary features
- Focused visualizations on top categories
- Recommended binning strategies for deployment

### Challenge 5: Feature Engineering Complexity
**Problem**: Need to create meaningful features without data leakage
**Solution**:
- Systematic feature engineering pipeline
- Careful temporal and logical feature construction
- Validation of feature relationships and distributions

## Technical Implementation

### Libraries and Dependencies
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy.stats
- **Machine Learning**: scikit-learn (multiple modules)
- **Data Loading**: kagglehub for dataset access
- **Environment Management**: warnings handling

### Enhanced Notebook Structure
1. **Project Header & Setup**: Student information and library imports
2. **Data Acquisition**: Kaggle dataset download and loading
3. **Initial Data Exploration**: Shape, types, basic statistics
4. **Data Quality Assessment**: Missing values, duplicates, outliers
5. **Data Cleaning**: Systematic duplicate removal and validation
6. **EDA Setup**: Visualization configuration and styling
7. **Numerical Features Analysis**: Distribution analysis with statistics
8. **Categorical Features Analysis**: Comprehensive categorical exploration
9. **Correlation Analysis**: Advanced correlation and relationship analysis
10. **Feature Engineering**: Comprehensive 32-feature creation pipeline
11. **Model Preparation**: Data splitting, scaling, and preprocessing
12. **Model Training**: Both classification and regression models
13. **Model Evaluation**: Comprehensive performance assessment
14. **Cross-Validation**: Rigorous generalization testing
15. **Business Insights**: Deployment readiness and recommendations

### Key Technical Improvements
- **Professional Visualization**: Consistent styling with statistical overlays
- **Automated Analysis**: Statistical interpretation and significance testing
- **Feature Engineering**: 32 new features with logical construction
- **Comprehensive Evaluation**: Multiple metrics with cross-validation
- **Business Integration**: Clear recommendations and insights

### Files
```
shoppers_behavior_analysis.ipynb    # Complete analysis notebook (15 cells)
README.md                           # Comprehensive project documentation
```
---

*This analysis represents a complete data mining pipeline from raw data to production-ready models
