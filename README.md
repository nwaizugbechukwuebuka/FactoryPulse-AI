# 🏭 FactoryPulse-AI

### AI-Powered Factory Operations Monitoring and Anomaly Intelligence

**FactoryPulse-AI** is a Python-based project that demonstrates how data analytics, machine learning, and API development can be combined to identify unusual factory operating measurements and translate detection results into understandable explanations, severity classifications, and recommended operational actions.

The project explores how intelligent monitoring systems can help operations teams identify deviations from expected behaviour and prioritize events for investigation.

**Project Status:** Phase 4 — Anomaly Intelligence Services Implemented
**Author:** Tobiloba Nwaizugbe
**Repository:** [FactoryPulse-AI](https://github.com/nwaizugbechukwuebuka/FactoryPulse-AI)

---

## 1. Business Problem

Manufacturing environments generate operational measurements that can deviate from expected values. Identifying these deviations quickly can help teams investigate potential equipment issues, abnormal operating conditions, and production inefficiencies.

However, identifying an unusual measurement is only part of the problem. Operations teams also need to understand why an event was flagged, assess its potential significance, and determine what to investigate next.

FactoryPulse-AI addresses this challenge through a prototype monitoring pipeline that combines rule-based detection, machine learning, evidence-based explanations, and severity-aware recommendations.

## 2. Project Objectives

The project was designed to demonstrate the ability to:

* Process and validate structured operational event data.
* Identify anomalous measurements using rule-based logic and machine learning.
* Compare detection approaches using quantitative evaluation metrics.
* Generate explanations based on available event measurements and detector results.
* Classify detected anomalies by severity.
* Recommend operational follow-up actions.
* Expose anomaly intelligence through a REST API.
* Validate implementation quality through automated tests and static analysis.

## 3. Key Features

### Data Engineering

Processes structured operational events and prepares data for anomaly detection. The development workflow uses synthetic data to support repeatable experiments and testing.

### Rule-Based Anomaly Detection

Applies predefined conditions to identify measurements that breach configured operating criteria.

### Machine Learning

Uses the Isolation Forest algorithm to identify unusual observations based on patterns in the available data.

### Model Evaluation

Compares detection approaches using precision, recall, F1-score, and false-positive rate. A held-out test dataset is used to evaluate detection performance.

### Anomaly Explanation

Generates structured explanations containing the detection method, available measurement evidence, and a human-readable summary.

### Severity Classification

Classifies anomalies into five levels:

* **None:** No anomaly was detected.
* **Low:** A detected anomaly has a measured deviation below the medium threshold.
* **Medium:** Measured deviation is at least 10%.
* **High:** Measured deviation is at least 25%.
* **Critical:** Measured deviation is at least 50%.

The thresholds are configurable in the severity-classification service. These categories are prototype decision-support labels, not validated industrial safety classifications.

### Operational Recommendations

Produces suggested follow-up actions based on the assigned severity, including continued monitoring, measurement verification, operator notification, escalation, and preservation of event data.

### REST API

Uses FastAPI to expose application health checks and an endpoint for requesting structured anomaly intelligence.

---

## 4. Machine Learning Evaluation

The anomaly detection component was evaluated using a synthetic dataset containing **600 events**, split into 450 training observations and 150 held-out test observations.

The following results were recorded for the Isolation Forest model:

| Metric              | Result |
| ------------------- | -----: |
| Precision           | 82.35% |
| Recall              | 87.50% |
| F1-score            | 84.85% |
| False-positive rate |  2.24% |
| True positives      |     14 |
| True negatives      |    131 |
| False positives     |      3 |
| False negatives     |      2 |

### What the Results Mean

* **Precision:** Approximately 82% of the events flagged by the model were actual anomalies according to the synthetic evaluation labels.
* **Recall:** The model identified 87.5% of the anomalies present in the held-out test dataset.
* **F1-score:** The model achieved a balance between precision and recall, with an F1-score of approximately 84.85%.
* **False-positive rate:** Approximately 2.24% of normal events in the test set were incorrectly flagged.

These results demonstrate a quantitative evaluation workflow rather than relying solely on visual inspection or anecdotal examples.

**Evaluation limitation:** These results come from synthetic data and do not establish performance on real factory equipment, production systems, or industrial safety scenarios. Performance in a real environment would require representative operational data, validated labels, and further testing.

---

## 5. Technical Architecture

The current implementation separates anomaly intelligence into focused services.

```text
                Synthetic Operational Events
                            |
                            v
                 Data Preparation and
                    Validation
                            |
                            v
                 Anomaly Detection
                  /             \
                 v               v
          Rule-Based         Isolation Forest
           Detection           Detection
                  \             /
                   v           v
                 Detection Results
                         |
                         v
               Anomaly Intelligence
                  /      |      \
                 v       v       v
           Explanation Severity Recommendations
                  \      |      /
                   v     v     v
                Structured API Response
                         |
                         v
                      FastAPI
```

The architecture separates detection from interpretation. This makes it easier to test individual services, compare detection methods, and extend the application without combining every responsibility in a single module.

**Implementation note:** The current intelligence API accepts detector results supplied in the request. Automatically executing the detection pipeline from an incoming event through to the final intelligence response remains a further integration step.

---

## 6. Technology Stack

| Technology     | Application                                       |
| -------------- | ------------------------------------------------- |
| Python         | Core application logic and service implementation |
| Pandas         | Structured data preparation and analysis          |
| Scikit-learn   | Machine learning and Isolation Forest             |
| FastAPI        | REST API development                              |
| Pydantic       | API request validation                            |
| Pytest         | Automated testing                                 |
| Ruff           | Python linting and code-quality checks            |
| Git and GitHub | Version control and source-code hosting           |

---

## 7. Project Structure

The main Phase 4 components are organized as follows:

```text
FactoryPulse-AI/
├── backend/
│   └── app/
│       ├── api/
│       │   └── routes/
│       │       └── intelligence.py
│       ├── services/
│       │   ├── anomaly_explanation.py
│       │   ├── anomaly_severity.py
│       │   ├── anomaly_recommendations.py
│       │   └── anomaly_intelligence.py
│       └── main.py
├── tests/
│   └── unit/
├── data/
│   └── processed/
├── scripts/
├── README.md
└── .gitignore
```

*This is a high-level overview of the relevant components, not an exhaustive listing of every file in the repository.*

### Core Services

| Component                    | Responsibility                                          |
| ---------------------------- | ------------------------------------------------------- |
| `anomaly_explanation.py`     | Produces evidence-based anomaly explanations            |
| `anomaly_severity.py`        | Assigns severity using configured thresholds            |
| `anomaly_recommendations.py` | Generates severity-specific operational recommendations |
| `anomaly_intelligence.py`    | Combines explanations, severity, and recommendations    |
| `intelligence.py`            | Exposes the intelligence service through FastAPI        |
| `main.py`                    | Configures the FastAPI application and health endpoints |

---

## 8. Running the Application

### Prerequisites

* Python 3.11
* Git
* A terminal such as Windows PowerShell

### Clone the Repository

```powershell
git clone https://github.com/nwaizugbechukwuebuka/FactoryPulse-AI.git
cd FactoryPulse-AI
```

### Create and Activate a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

Install the project's declared dependencies using the installation method supported by the repository.

If the project provides a development dependency group, use:

```powershell
python -m pip install -e ".[dev]"
```

Otherwise, follow the dependency installation instructions defined by the project's configuration files.

### Start the API

With the environment configured, run:

```powershell
python -m uvicorn backend.app.main:app --reload
```

The local application should be available at:

* **Interactive API documentation:** http://127.0.0.1:8000/docs
* **Health check:** http://127.0.0.1:8000/health
* **Readiness check:** http://127.0.0.1:8000/ready

The interactive documentation allows users to inspect the available endpoints and submit test requests.

---

## 9. Example API Request

The anomaly intelligence endpoint is:

`POST /api/v1/intelligence/analyze`

Example request:

```json
{
  "event": {
    "event_id": "EVT-001",
    "event_type": "temperature",
    "value": 145,
    "target_value": 100
  },
  "is_anomaly": true,
  "detection_method": "hybrid",
  "rule_result": {
    "is_anomaly": true,
    "reason": "Temperature exceeds the configured target."
  },
  "ml_prediction": 1
}
```

The service combines the supplied event and detection results to return a structured response containing:

* Event identification and anomaly status.
* An explanation and available measurement evidence.
* A severity classification.
* A severity rationale.
* Recommended operational actions.

For this example, the measured value is 45% above the target. Under the current prototype thresholds, that corresponds to **high severity**.

This example illustrates the intelligence service's current interface; it does not represent a live industrial alert or an automatically executed detection pipeline.

---

## 10. Testing and Code Quality

Automated testing and static analysis are integral to the development workflow.

The latest recorded verification results are:

| Check                  | Result                                     |
| ---------------------- | ------------------------------------------ |
| Complete Pytest suite  | 75 tests passed                            |
| Ruff static analysis   | All checks passed                          |
| Git branch             | `main`                                     |
| GitHub synchronization | Local branch up to date with `origin/main` |

To run the tests:

```powershell
python -m pytest -q
```

To run the code-quality checks:

```powershell
python -m ruff check .
```

These results reflect the project state at the latest verification. They should be rerun after subsequent changes.

---

## 11. Current Scope and Limitations

FactoryPulse-AI is a development prototype built around synthetic operational data.

The current implementation demonstrates anomaly detection, quantitative evaluation, anomaly intelligence services, and API exposure. It should not be interpreted as a deployed industrial monitoring platform.

The following areas require further development or validation:

* End-to-end integration of incoming events with the actual detection pipeline.
* Evaluation against representative real-world industrial measurements.
* Validation of anomaly thresholds and severity classifications with operational specialists.
* Production-grade monitoring, logging, authentication, and deployment.
* Integration with actual factory systems and operational data sources.

The prototype does not directly connect to Tesla's systems or any other live manufacturing environment.

---

## 12. Future Development

Potential next steps include:

* Connecting the API to the actual Phase 3 detection pipeline.
* Adding end-to-end integration tests.
* Evaluating model performance on larger and more representative datasets.
* Improving event correlation and investigation context.
* Adding a monitoring dashboard and operational reporting.
* Strengthening API security, observability, and deployment readiness.

These are planned extensions rather than claims about completed functionality.

---

## 13. Skills Demonstrated

**Python Development**

* Modular service design
* Input validation and error handling
* API implementation

**Data Analytics and Machine Learning**

* Structured data preparation
* Anomaly detection
* Model evaluation using precision, recall, F1-score, and false-positive rate

**Software Quality Engineering**

* Automated unit testing with Pytest
* Static analysis with Ruff
* Validation of edge cases and invalid inputs

**AI-Driven Operational Decision Support**

* Evidence-based explanations
* Severity classification
* Structured recommendations

**Engineering Workflow**

* Version control with Git
* GitHub repository management
* Incremental implementation and verification

---

## Summary

FactoryPulse-AI demonstrates how Python, machine learning, data analytics, and API engineering can be combined to develop an anomaly-monitoring prototype.

The project goes beyond identifying unusual measurements by adding structured explanations, severity classification, and operational recommendations. Its quantitative evaluation and automated tests demonstrate a measurable, quality-focused engineering approach.

The current implementation provides a foundation for further work on integrated anomaly detection, operational intelligence, and production-oriented monitoring systems.

**Project:** FactoryPulse-AI
**Category:** Machine Learning | Data Analytics | Python Engineering | Industrial Operations Intelligence
**Status:** Prototype — Phase 4 implemented and tested
[![GitHub](https://img.shields.io/badge/GitHub-Project-181717.svg?style=flat\&logo=github)](https://github.com/nwaizugbechukwuebuka/FactoryPulse-AI)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077b5.svg?style=flat\&logo=linkedin)](https://www.linkedin.com/in/chukwuebuka-tobiloba-nwaizugbe/)