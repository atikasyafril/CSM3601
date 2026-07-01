/**
 * predictor.js  (Random Forest)
 * ----------------------------------------
 * Pure JavaScript re-implementation of the trained scikit-learn pipeline
 * matching CSM3601_Heart_Disease_Prediction.ipynb Phase 2 preprocessing:
 *   pd.get_dummies(drop_first=True) -> MinMaxScaler (numeric only) ->
 *   SelectKBest(k=10) -> Random Forest (majority vote of 100 trees)
 *
 * All numbers come from model/model_artifact.json (exported by python/train_model.py).
 * No Python needed at runtime.
 */

const fs   = require("fs");
const path = require("path");

const ARTIFACT_PATH = path.join(__dirname, "..", "model", "model_artifact.json");
let artifact = null;

function loadArtifact() {
  if (!fs.existsSync(ARTIFACT_PATH)) return null;
  artifact = JSON.parse(fs.readFileSync(ARTIFACT_PATH, "utf-8"));
  return artifact;
}

function isModelLoaded() { return artifact !== null; }
function getMetadata()   { return artifact; }

function traverseTree(node, features) {
  if (node.leaf) return node.prediction;
  const val = features[node.feature] !== undefined ? features[node.feature] : 0;
  return val <= node.threshold
    ? traverseTree(node.left,  features)
    : traverseTree(node.right, features);
}

function predict(input) {
  if (!artifact) throw new Error("Model artifact not loaded. Run python/train_model.py first.");

  const { numeric_cols, categorical_cols, scaler_stats, selected_features, rf_trees } = artifact;

  const encoded = {};

  // 1. MinMaxScaler on numeric cols
  numeric_cols.forEach((col) => {
    const { data_min, data_max } = scaler_stats[col];
    const x = Number(input[col]);
    const range = data_max - data_min;
    encoded[col] = range === 0 ? 0 : (x - data_min) / range;
  });

  // 2. Passthrough binary cols
  ["sex", "fbs", "exang", "ca"].forEach((col) => {
    encoded[col] = Number(input[col]);
  });

  // 3. One-hot encode categorical cols (drop_first=True, sorted string order)
  const catCategories = {
    cp:      ["0","1","2","3"],
    restecg: ["0","1","2"],
    slope:   ["0","1","2"],
    thal:    ["0","1","2"],
  };
  categorical_cols.forEach((col) => {
    const cats = catCategories[col];
    const val  = String(input[col]);
    cats.slice(1).forEach((cat) => {
      encoded[`${col}_${cat}`] = val === cat ? 1 : 0;
    });
  });

  // 4. Build selected-feature vector
  const featVec = {};
  selected_features.forEach((f) => {
    featVec[f] = encoded[f] !== undefined ? encoded[f] : 0;
  });

  // 5. Random Forest majority vote
  let votes = 0;
  rf_trees.forEach((tree) => {
    votes += traverseTree(tree, featVec);
  });

  const probability = votes / rf_trees.length;
  const prediction  = probability >= 0.5 ? 1 : 0;

  return { prediction, probability };
}

module.exports = { loadArtifact, isModelLoaded, getMetadata, predict };
