/**
 * predictor.js
 * -------------
 * Pure JavaScript re-implementation of the trained scikit-learn pipeline:
 *   MinMaxScaler -> OneHotEncoder(drop="first") -> SelectKBest -> LogisticRegression
 *
 * All the numbers (scaler min/max, one-hot categories, selected features,
 * logistic regression coefficients + intercept) come from model/model_artifact.json,
 * which was exported once by python/train_model.py.
 *
 * No Python is needed while the website is running - this file does the
 * exact same math by hand.
 */

const fs = require("fs");
const path = require("path");

const ARTIFACT_PATH = path.join(__dirname, "..", "model", "model_artifact.json");

let artifact = null;

function loadArtifact() {
  if (!fs.existsSync(ARTIFACT_PATH)) {
    return null;
  }
  const raw = fs.readFileSync(ARTIFACT_PATH, "utf-8");
  artifact = JSON.parse(raw);
  return artifact;
}

function isModelLoaded() {
  return artifact !== null;
}

function getMetadata() {
  return artifact;
}

/**
 * input: plain object keyed by raw feature names, e.g.
 *   { age: 63, sex: 1, cp: 0, trestbps: 145, chol: 233, fbs: 1,
 *     restecg: 2, thalach: 150, exang: 0, oldpeak: 2.3, slope: 2, ca: 0, thal: 2 }
 *
 * returns: { prediction: 0|1, probability: 0..1 }
 */
function predict(input) {
  if (!artifact) {
    throw new Error("Model artifact not loaded. Run python/train_model.py first.");
  }

  const {
    numeric_cols,
    scaler,
    categorical_cols,
    categories,
    passthrough_cols,
    encoded_feature_names,
    selected_feature_names,
    coefficients,
    intercept,
  } = artifact;

  // Build a map of every encoded feature -> its computed value
  const encodedValues = {};

  // 1. Numeric features: MinMaxScaler -> (x - min) / (max - min)
  numeric_cols.forEach((col, i) => {
    const min = scaler.data_min[i];
    const max = scaler.data_max[i];
    const x = Number(input[col]);
    const range = max - min;
    encodedValues[col] = range === 0 ? 0 : (x - min) / range;
  });

  // 2. Categorical features: One-Hot Encoding with drop="first"
  categorical_cols.forEach((col, idx) => {
    const colInfo = categories[idx];
    const rawValue = String(input[col]);
    colInfo.kept_categories.forEach((cat) => {
      const featureName = `${col}_${cat}`;
      encodedValues[featureName] = rawValue === String(cat) ? 1 : 0;
    });
  });

  // 3. Passthrough features (sex, fbs, exang, ca) - used as-is
  passthrough_cols.forEach((col) => {
    encodedValues[col] = Number(input[col]);
  });

  // 4. Build the SAME ordered vector the SelectKBest step picked, then
  //    apply the logistic regression: z = w . x + b
  let z = intercept;
  selected_feature_names.forEach((featureName, i) => {
    const value = encodedValues[featureName] !== undefined ? encodedValues[featureName] : 0;
    z += coefficients[i] * value;
  });

  const probability = 1 / (1 + Math.exp(-z));
  const prediction = probability >= 0.5 ? 1 : 0;

  return { prediction, probability };
}

module.exports = { loadArtifact, isModelLoaded, getMetadata, predict };
