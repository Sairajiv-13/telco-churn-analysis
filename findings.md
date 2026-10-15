# Findings: who churns, and what to do about it

Dataset: IBM Telco sample, 7,043 customers. 1,869 churned, an overall churn
rate of 26.5 percent. Segment rates below are computed on the full dataset.
Model numbers come from a held-out test set of 1,761 customers (stratified
25 percent split, seed 42) and match reports/metrics.json.

## Churn is a first-year problem

Almost half of first-year customers leave. Churn by tenure band:

| Tenure (months) | Customers | Churn rate |
| --- | --- | --- |
| 0 to 12 | 2,186 | 47.4% |
| 13 to 24 | 1,024 | 28.7% |
| 25 to 48 | 1,594 | 20.4% |
| 49 to 72 | 2,239 | 9.5% |

First-year customers are 31.0 percent of the base but 55.5 percent of all
churners (1,037 of 1,869). The median tenure of a customer who churned is
10 months; for a customer who stayed it is 38 months. Tenure is also the
strongest model driver by a wide margin (permutation importance 0.162 of
AUC, against 0.046 for the next feature), and SHAP on the fitted model
ranks it first as well.

## Contract type is the biggest controllable split

| Contract | Customers | Churn rate |
| --- | --- | --- |
| Month-to-month | 3,875 | 42.7% |
| One year | 1,473 | 11.3% |
| Two year | 1,695 | 2.8% |

A month-to-month customer churns at about 15 times the rate of a two-year
customer. Some of this is self-selection: people who expect to stay sign
longer contracts. But the size of the gap, across 3,875 customers, makes
contract conversion the clearest retention lever in the data.

## How a customer pays tracks how long they stay

| Payment method | Customers | Churn rate |
| --- | --- | --- |
| Electronic check | 2,365 | 45.3% |
| Mailed check | 1,612 | 19.1% |
| Bank transfer (automatic) | 1,544 | 16.7% |
| Credit card (automatic) | 1,522 | 15.2% |

The two automatic methods sit at 15 to 17 percent. Electronic check, the
one method that makes the customer act every month, sits at 45.3 percent.
Paperless billing shows the same shape in reverse of what most teams
expect: customers on paperless billing churn at 33.6 percent against
16.3 percent for paper bills, most likely because paperless skews toward
month-to-month electronic check accounts.

The worst combined segment in the dataset is month-to-month customers
paying by electronic check: 1,850 customers churning at 53.7 percent.
First-year month-to-month customers churn at 51.4 percent (1,994
customers). These two groups overlap heavily and are the natural first
target list.

## Fiber customers leave, and unsupported accounts leave faster

Fiber optic customers churn at 41.9 percent (3,096 customers) against
19.0 percent for DSL. Among internet customers, the add-on split is
sharp:

| Add-on | Without it | With it |
| --- | --- | --- |
| Tech support | 41.6% (3,473) | 15.2% (2,044) |
| Online security | 41.8% (3,498) | 14.6% (2,019) |

Read this as an association, not proof of cause. Customers who buy
support may be more committed to begin with. The honest way to use the
number is a pilot: offer the add-ons to a slice of unsupported fiber
accounts and measure whether their churn moves.

Price plays a smaller, real role. Customers who churned paid $74.44 a
month on average against $61.27 for customers who stayed, and accounts
above the median bill ($70.35) churn at 35.2 percent against 17.9
percent at or below it. Monthly charges rank fourth in both importance
methods (permutation 0.033, SHAP fourth), behind tenure, internet
service, and contract.

## The model is good enough to aim outreach, not to replace judgment

Two models were trained on the same split, with class weights instead of
resampling so the probabilities stay comparable:

| Model | ROC AUC | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.846 | 0.750 | 0.519 | 0.797 | 0.628 |
| Random forest | 0.839 | 0.786 | 0.593 | 0.619 | 0.606 |

The random forest is tidier (fewer false alarms) but misses far more
churners. For retention work a missed churner costs more than a wasted
offer, so the logistic regression ships, chosen on ROC AUC. Threshold
choice matters more than the model choice:

| Threshold | Precision | Recall |
| --- | --- | --- |
| 0.30 | 0.433 | 0.927 |
| 0.50 | 0.519 | 0.797 |
| 0.70 | 0.610 | 0.600 |

At a 0.30 working threshold the model catches 92.7 percent of churners,
and about 43 percent of the flagged accounts are true churners. That is
a workable outreach list. At 0.70 the list is cleaner and misses 40
percent of the people who leave.

Permutation importance (drop in test ROC AUC when a column is shuffled):
tenure 0.162, InternetService 0.046, Contract 0.043, MonthlyCharges
0.033, TotalCharges 0.012, then a long tail under 0.006. SHAP values on
the same model put the same four features on top in the same order, so
the driver story does not depend on which method you trust.

## Actions

1. **Build the save program around the first year.** Churn is 47.4
   percent in months 0 to 12 and falls every band after that. Score
   accounts monthly, work the list at the 0.30 threshold, and put the
   human effort into months 3 to 12, before the median churner leaves at
   month 10.
2. **Convert month-to-month accounts to term contracts.** A one-year
   offer aimed at high-scoring month-to-month customers attacks the
   42.7 versus 2.8 percent contract gap. Even partial conversion beats
   discounting the whole base.
3. **Move electronic check customers to automatic payment.** 45.3
   percent churn against 15 to 17 percent for automatic methods. Ask at
   signup, ask again in the first bill cycle, and start with the 1,850
   month-to-month electronic check accounts churning at 53.7 percent.
4. **Pilot support and security add-ons on unsupported fiber accounts.**
   Fiber churns at 41.9 percent, and internet customers without tech
   support churn at 41.6 percent against 15.2 percent with it. Treat it
   as a test with a holdout group, since the gap may be selection
   rather than the add-on itself.
5. **Review plan fit for high-bill accounts.** Customers above the
   median bill churn at about twice the rate of those below it, and
   churners pay roughly $13 more a month on average. A plan-fit check
   for high-charge, high-score accounts is cheaper than winning them
   back.

## Caveats

This is a sample dataset published by IBM for analysis, with one row per
customer and no time series of behavior. Every segment rate here is an
association in a snapshot. Contract, payment method, and add-on effects
are partly self-selection, and only a controlled pilot separates cause
from correlation. The model ranks risk well (AUC 0.846); it does not
explain why any individual customer leaves.
