const functions = require("firebase-functions");
const express = require("express");
const path = require("path");
const predictor = require("./lib/predictor");

const app = express();

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

predictor.loadArtifact();

const FIELD_ORDER = [
  "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
  "thalach", "exang", "oldpeak", "slope", "ca", "thal",
];
const FIELD_LABELS = {
  age: "Age (years)", sex: "Sex", cp: "Chest Pain Type",
  trestbps: "Resting Blood Pressure (mm Hg)", chol: "Serum Cholesterol (mg/dl)",
  fbs: "Fasting Blood Sugar > 120 mg/dl", restecg: "Resting ECG Result",
  thalach: "Max Heart Rate Achieved", exang: "Exercise Induced Angina",
  oldpeak: "ST Depression (oldpeak)", slope: "Slope of Peak Exercise ST Segment",
  ca: "Number of Major Vessels (0-3)", thal: "Thalassemia",
};
const FIELD_OPTIONS = {
  sex: [[1, "Male"], [0, "Female"]],
  cp: [[0, "Typical Angina"], [1, "Atypical Angina"], [2, "Non-anginal Pain"], [3, "Asymptomatic"]],
  fbs: [[1, "Yes (> 120 mg/dl)"], [0, "No"]],
  restecg: [[0, "Normal"], [1, "ST-T Wave Abnormality"], [2, "Left Ventricular Hypertrophy"]],
  exang: [[1, "Yes"], [0, "No"]],
  slope: [[0, "Upsloping"], [1, "Flat"], [2, "Downsloping"]],
  ca: [[0, "0"], [1, "1"], [2, "2"], [3, "3"]],
  thal: [[0, "Normal"], [1, "Fixed Defect"], [2, "Reversible Defect"]],
};
const NUMERIC_FIELDS = ["age", "trestbps", "chol", "thalach", "oldpeak"];

app.get("/", (req, res) => {
  res.render("index", {
    fieldOrder: FIELD_ORDER, fieldLabels: FIELD_LABELS, fieldOptions: FIELD_OPTIONS,
    numericFields: NUMERIC_FIELDS, modelLoaded: predictor.isModelLoaded(), error: null,
  });
});

app.post("/predict", (req, res) => {
  if (!predictor.isModelLoaded()) return res.redirect("/");
  const values = {};
  let valid = true;
  FIELD_ORDER.forEach((field) => {
    const raw = req.body[field];
    const num = Number(raw);
    if (raw === undefined || raw === "" || Number.isNaN(num)) valid = false;
    values[field] = num;
  });
  if (!valid) {
    return res.render("index", {
      fieldOrder: FIELD_ORDER, fieldLabels: FIELD_LABELS, fieldOptions: FIELD_OPTIONS,
      numericFields: NUMERIC_FIELDS, modelLoaded: predictor.isModelLoaded(),
      error: "Please fill in every field with a valid value.",
    });
  }
  const { prediction, probability } = predictor.predict(values);
  const meta = predictor.getMetadata();
  res.render("result", {
    prediction, probability: (probability * 100).toFixed(1),
    modelName: meta.model_name, values, fieldLabels: FIELD_LABELS,
  });
});

app.post("/api/predict", (req, res) => {
  if (!predictor.isModelLoaded()) {
    return res.status(503).json({ error: "Model not loaded." });
  }
  try {
    res.json(predictor.predict(req.body));
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.get("/about", (req, res) => {
  const meta = predictor.getMetadata();
  res.render("about", {
    metrics: meta ? meta.metrics_all_models : [],
    modelName: meta ? meta.model_name : "N/A",
    selectedFeatures: meta ? meta.selected_feature_names : [],
    modelLoaded: predictor.isModelLoaded(),
  });
});

exports.app = functions.https.onRequest(app);
