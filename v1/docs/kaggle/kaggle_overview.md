Title: Real Estate Demand Prediction

URL Source: https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview

Markdown Content:
Real Estate Demand Prediction | Kaggle

===============

menu

[Skip to content](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview#site-content)

[![Image 3: Kaggle](https://www.kaggle.com/static/images/site-logo.svg)](https://www.kaggle.com/)

Create

search​

*   [explore Home](https://www.kaggle.com/) 
*   [emoji_events Competitions](https://www.kaggle.com/competitions) 
*   [table_chart Datasets](https://www.kaggle.com/datasets) 
*   [tenancy Models](https://www.kaggle.com/models) 
*   [leaderboard Benchmarks](https://www.kaggle.com/benchmarks) 
*   [code Code](https://www.kaggle.com/code) 
*   [comment Discussions](https://www.kaggle.com/discussions) 
*   [school Learn](https://www.kaggle.com/learn) 

*   [expand_more More](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview#) 

auto_awesome_motion

View Active Events

menu

[Skip to content](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview#site-content)

[![Image 4: Kaggle](https://www.kaggle.com/static/images/site-logo.svg)](https://www.kaggle.com/)

search​

[Sign In](https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2Fcompetitions%2Fchina-real-estate-demand-prediction%2Foverview)

[Register](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2Fcompetitions%2Fchina-real-estate-demand-prediction%2Foverview)

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.

[Learn more](https://www.kaggle.com/cookies)

OK, Got it.

[](https://www.kaggle.com/lazylo)Lazylo  · Community Prediction Competition · a month to go

Join Competition

more_horiz

Real Estate Demand Prediction
=============================

China's first real-estate demand prediction challenge, developing models to forecast housing demand and guide real investment decisions.

![Image 5](https://www.kaggle.com/competitions/111876/images/header)

Real Estate Demand Prediction
-----------------------------

[Overview](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview)[Data](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/data)[Code](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/code)[Models](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/models)[Discussion](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/discussion)[Leaderboard](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/leaderboard)[Rules](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/rules)

Overview
--------

Welcome to the Real Estate Demand Prediction Challenge! We are inviting the Kaggle community to participate in the first Kaggle competition that focuses on China's real estate market and help forecast monthly residential demand in China.

We look forward to seeing how you apply your data science expertise to help us shape the future of real estate!

Start

a month ago

###### Close

a month to go

### Description

link

keyboard_arrow_up

In China’s fast-evolving and highly dynamic housing market, accurately forecasting residential demand is vital for investment and development decisions. This competition challenges you to develop a machine learning model that predicts each sector's monthly sales for newly launched private residential projects, using historical transaction data, market conditions, and other relevant features.

Why Participate:
================

*   Real-World Impact: Your model will influence investment decisions and future development strategies, helping shape China's housing market.

*   Networking Opportunities: Engage with industry professionals and fellow Kaggle participants. Winners or excellent participants may be invited into the AI panel of a world renowned real estate group.

*   Prizes and Recognition: Win exciting prizes and showcase your skills to a global audience. This is your opportunity to tackle a real-world business problem and showcase your skills in data science and predictive modeling. Your insights will help shape the future of urban living in one of the world's most vibrant cities.

How to Get Started:
===================

*   Review the provided data and competition details.

*   Build and refine your predictive model.

*   Submit your predictions.

We look forward to seeing how you apply your data science expertise to help us shape the future of real estate!

### Evaluation

link

keyboard_arrow_up

Submissions will be evaluated using a custom two-stage metric based on the absolute percentage error between predicted and actual values. The custom score is computed from a scaled [MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) with two stages.

**First stage:**

If over 30% of the submitted samples have absolute percentage errors exceeding 100%, a score of 0 is immediately given.

![Image 6: score_0.png](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28096579%2F573450f7364d161293c2ce4e3c7e4235%2Fscore_0.png?generation=1754638731130078&alt=media)

**Second stage:**

Otherwise, an MAPE is calculated with the samples that have absolute percentage errors less than or equal to 1. Then the MAPE is divided by the fraction of absolute percentage errors less than or equal to 1. Finally, the score is 1 minus the scaled MAPE.

![Image 7: score_1.png](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28096579%2Fca4b376cf10f63393146e5f6d9ced5cc%2Fscore_1.png?generation=1754638744628468&alt=media)

Submission File
---------------

The submission file must contain exactly these two columns: `id` and `new_house_transaction_amount`. Each id is formatted as a month in the form of `%Y %b` and a sector id in the form of `sector n` joined by an underscore. For each id in the test set, you must predict the total new house transaction amount in ten thousand Chinese Yuan for the specific month and sector. The total new house transaction amount is equal to total new house transaction area times transaction price per area. The file should contain a header and have the following format:

```
id,new_house_transaction_amount
2024 Aug_sector 1,49921.00868
2024 Aug_sector 2,92356.22985
2024 Aug_sector 3,480269.7819
etc.
```

content_copy

### Submission Requirements

*   **Preserve row order**: Maintain the exact same row order as in test.csv
*   **Exact Columns**: The submission file should contain the exact columns as in test.csv

### Prizes

link

keyboard_arrow_up

Top three participants/teams will win prizes with potential bonus.

**Total Prizes Available: $10,000 USD**

*   **First Prize(1 winner)- $2,500 USD or $5,000 USD with bonus**
*   **Second Prize(1 winner) - $1,500 USD or $3,000 USD with bonus**
*   **Third Prize(1 winner) - $1,000 USD or $2,000 USD with bonus**

Bonus Performance Threshold
---------------------------

Winners who achieve final score ≥ 0.75 receive a bonus by doubling their prize amount.

Winner Selection
----------------

**Additional Verification:** Private leaderboard metrics are not final. Organizers will request code from top-10 participants for final evaluation. Winning models may be incorporated into a real estate company's proprietary forecasting system for internal use.

**New Data Testing:** Organizers reserve the right to test candidate solutions on unpublished 2025 data.

### Citation

link

keyboard_arrow_up

Lazylo. Real Estate Demand Prediction. https://kaggle.com/competitions/china-real-estate-demand-prediction, 2025. Kaggle.

Cite

Competition Host
----------------

Lazylo

[](https://www.kaggle.com/lazylo)

Prizes & Awards
---------------

$10,000

Does not award Points or Medals

Participation
-------------

1,549 Entrants

380 Participants

361 Teams

4,471 Submissions

Tags
----

Custom Metric

Table of Contents

collapse_all

[Overview](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview/abstract)[Description](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview/description)[Evaluation](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview/evaluation)[Prizes](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview/prizes)[Citation](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/overview/citation)
