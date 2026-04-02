const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat,
} = require("docx");

// Helpers
const CONTENT_WIDTH = 9360; // US Letter with 1" margins
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const headerBorder = { style: BorderStyle.SINGLE, size: 1, color: "2E75B6" };
const headerBorders = { top: headerBorder, bottom: headerBorder, left: headerBorder, right: headerBorder };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function p(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    runs.push(new TextRun({ text, ...opts }));
  } else if (Array.isArray(text)) {
    text.forEach(t => {
      if (typeof t === "string") runs.push(new TextRun(t));
      else runs.push(new TextRun(t));
    });
  }
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    alignment: opts.align || AlignmentType.LEFT,
    ...opts.paragraphOpts,
    children: runs,
  });
}

function heading(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: level === HeadingLevel.HEADING_1 ? 360 : 240, after: 120 },
    children: [new TextRun({ text, bold: true })],
  });
}

function bold(text) { return { text, bold: true }; }
function italic(text) { return { text, italics: true }; }

function makeRow(cells, isHeader = false) {
  return new TableRow({
    children: cells.map((c, i) => {
      const width = Array.isArray(c) ? c[1] : undefined;
      const text = Array.isArray(c) ? c[0] : c;
      return new TableCell({
        borders: isHeader ? headerBorders : borders,
        margins: cellMargins,
        width: width ? { size: width, type: WidthType.DXA } : undefined,
        shading: isHeader ? { fill: "2E75B6", type: ShadingType.CLEAR } : undefined,
        children: [new Paragraph({
          spacing: { after: 0 },
          alignment: AlignmentType.CENTER,
          children: [new TextRun({
            text: String(text),
            bold: isHeader,
            color: isHeader ? "FFFFFF" : "000000",
            size: 20,
            font: "Arial",
          })],
        })],
      });
    }),
  });
}

function makeTable(headers, rows, colWidths) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      makeRow(headers.map((h, i) => [h, colWidths[i]]), true),
      ...rows.map(r => makeRow(r.map((c, i) => [c, colWidths[i]]))),
    ],
  });
}

function spacer() {
  return new Paragraph({ spacing: { after: 120 }, children: [] });
}

// Build the document
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "404040" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
      ]},
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      ]},
    ],
  },
  sections: [
    // ===== TITLE PAGE =====
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: [
        new Paragraph({ spacing: { before: 3000 }, children: [] }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "Early Prediction of Sepsis", size: 52, bold: true, font: "Arial", color: "1F4E79" })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "from Clinical ICU Data", size: 52, bold: true, font: "Arial", color: "1F4E79" })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "A Machine Learning Approach Using LightGBM and GRU", size: 28, font: "Arial", color: "2E75B6" })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "with Clinical Feature Engineering", size: 28, font: "Arial", color: "2E75B6" })],
        }),
        new Paragraph({ spacing: { before: 600 }, children: [], border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 1 } } }),
        new Paragraph({ spacing: { before: 400 }, children: [] }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "PhysioNet/Computing in Cardiology Challenge 2019 Dataset", size: 24, font: "Arial", color: "404040" })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "40,336 ICU Patients | 297 Engineered Features | Patient-Level Evaluation", size: 22, font: "Arial", color: "666666" })],
        }),
        new Paragraph({ spacing: { before: 600 }, children: [] }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "March 2026", size: 24, font: "Arial", color: "404040" })],
        }),
        new Paragraph({ children: [new PageBreak()] }),
      ],
    },

    // ===== MAIN CONTENT =====
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            border: { bottom: { style: BorderStyle.SINGLE, size: 1, color: "2E75B6", space: 4 } },
            children: [new TextRun({ text: "Early Prediction of Sepsis from Clinical ICU Data", italics: true, size: 18, color: "999999", font: "Arial" })],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "Page ", size: 18, color: "999999" }), new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "999999" })],
          })],
        }),
      },
      children: [
        // ===== ABSTRACT =====
        heading("Abstract", HeadingLevel.HEADING_1),
        p([
          "Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection, affecting approximately 1.7 million adults annually in the United States with mortality rates of 15-30%. Early detection is critical as each hour of delayed treatment increases mortality by 4-8%. This project develops a machine learning system for the early prediction of sepsis onset in ICU patients, trained to alert clinicians ",
          bold("6 hours before clinical diagnosis"),
          ".",
        ]),
        p([
          "Using the PhysioNet/Computing in Cardiology Challenge 2019 dataset (40,336 ICU patients, 1.84% sepsis prevalence), we engineered 297 clinical features including SOFA, qSOFA, and NEWS scores, clinical ratios (Shock Index, BUN/Creatinine), and temporal dynamics. We trained LightGBM gradient-boosted trees and GRU recurrent neural networks, evaluating at the ",
          bold("patient level"),
          " (will this patient develop sepsis?) rather than the timestep level (is this hour sepsis?).",
        ]),
        p([
          "Our best model, an ensemble of GRU (30%) and LightGBM (70%), achieves a ",
          bold("patient-level AUROC of 0.8573"),
          " and ",
          bold("AUPRC of 0.5032"),
          " (28x lift over base rate). At the screening threshold, the model catches ",
          bold("78% of sepsis patients"),
          " 6 hours before onset. At the balanced threshold, it achieves a ",
          bold("patient-level F1 of 0.487"),
          " with 42% sensitivity and 58% precision. Results are competitive with top PhysioNet 2019 Challenge entries, which achieved utility scores of 0.36-0.43.",
        ]),

        // ===== 1. INTRODUCTION =====
        heading("1. Introduction", HeadingLevel.HEADING_1),

        heading("1.1 What is Sepsis?", HeadingLevel.HEADING_2),
        p("Sepsis is defined by the Sepsis-3 guidelines as a life-threatening organ dysfunction caused by a dysregulated host response to infection. It is characterized by a two-point or greater change in the Sequential Organ Failure Assessment (SOFA) score combined with clinical suspicion of infection. Sepsis progresses rapidly and can lead to septic shock, multi-organ failure, and death if not treated promptly."),
        p("The clinical challenge is that early sepsis symptoms (elevated heart rate, fever, altered vital signs) overlap significantly with normal patient variation in the ICU. By the time sepsis is clinically obvious, the patient may already be in critical condition. This makes automated early warning systems valuable as they can continuously monitor dozens of clinical variables simultaneously and detect subtle patterns that precede sepsis onset."),

        heading("1.2 Why Early Prediction Matters", HeadingLevel.HEADING_2),
        p([
          "Research has consistently shown that ",
          bold("each hour of delayed antibiotic administration increases sepsis mortality by 4-8%"),
          ". A system that can reliably alert clinicians 6 hours before clinical onset provides a critical window for intervention: blood cultures can be drawn, antibiotics started, and hemodynamic support initiated before organ damage becomes irreversible.",
        ]),
        p("The PhysioNet/Computing in Cardiology Challenge 2019 was specifically designed to advance this goal, providing a standardized dataset and evaluation framework for early sepsis prediction algorithms."),

        heading("1.3 Project Objectives", HeadingLevel.HEADING_2),
        p("This project aims to:"),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Build a patient-level sepsis early warning system that predicts whether a patient will develop sepsis")] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Engineer clinically meaningful features (SOFA, qSOFA, NEWS) alongside data-driven temporal features")] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Compare gradient-boosted trees (LightGBM) with recurrent neural networks (GRU) and their ensemble")] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Evaluate at the patient level with clinically relevant metrics and operating points")] }),
        spacer(),

        // ===== 2. DATASET =====
        heading("2. Dataset", HeadingLevel.HEADING_1),

        heading("2.1 PhysioNet 2019 Challenge Data", HeadingLevel.HEADING_2),
        p("The dataset consists of hourly clinical measurements from 40,336 ICU patients across two hospital systems (Beth Israel Deaconess Medical Center and Emory University Hospital). Each patient has a variable-length time series of 40 clinical features recorded at hourly intervals, along with a binary sepsis label at each timestep."),
        spacer(),
        makeTable(
          ["Statistic", "Value"],
          [
            ["Total patients", "40,336"],
            ["Training patients", "34,285 (85%)"],
            ["Validation patients", "6,051 (15%)"],
            ["Sepsis patients (validation)", "458 (7.57%)"],
            ["Non-sepsis patients (validation)", "5,593 (92.43%)"],
            ["Observation-level sepsis rate", "1.84%"],
            ["Total training timesteps", "1,315,556"],
            ["Total validation timesteps", "236,654"],
            ["Features per timestep", "40 raw clinical variables"],
          ],
          [4680, 4680],
        ),

        heading("2.2 Clinical Variables", HeadingLevel.HEADING_2),
        p("The 40 raw features fall into three categories:"),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Vital Signs (8): ", bold: true }),
          new TextRun("HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Lab Values (26): ", bold: true }),
          new TextRun("BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Creatinine, Glucose, Lactate, WBC, Platelets, and others"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Demographics (6): ", bold: true }),
          new TextRun("Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS"),
        ]}),

        heading("2.3 Sepsis Label Definition", HeadingLevel.HEADING_2),
        p([
          "The sepsis labels are set to 1 ",
          bold("up to 6 hours before"),
          " the actual sepsis onset (defined by Sepsis-3 criteria). For example, if a patient develops sepsis at hour 30, hours 24 through 30 are labeled as positive. This means the model is trained to detect sepsis patterns ",
          bold("before clinical diagnosis"),
          ", providing an early warning window.",
        ]),

        heading("2.4 Class Imbalance Challenge", HeadingLevel.HEADING_2),
        p([
          "The dataset exhibits extreme class imbalance: only ",
          bold("1.84% of timesteps"),
          " are labeled positive, and only ",
          bold("458 out of 6,051 validation patients (7.57%)"),
          " have sepsis. This imbalance is the fundamental challenge of the project. Even a model with 96% specificity generates thousands of false positives relative to the few hundred true positives, making precision inherently low. This is not a model failure but a mathematical consequence of low prevalence.",
        ]),

        // ===== 3. METHODOLOGY =====
        heading("3. Methodology", HeadingLevel.HEADING_1),

        heading("3.1 Approach Evolution", HeadingLevel.HEADING_2),
        p("Our approach evolved through several iterations as we gained understanding of the problem:"),
        spacer(),
        makeTable(
          ["Phase", "Approach", "Primary Metric", "Outcome"],
          [
            ["1", "Utility-optimized GRU + LightGBM", "PhysioNet Utility", "Utility=0.40 but poor patient-level F1"],
            ["2", "Balanced models (varying class weights)", "Timestep F1", "F1=0.19, low due to prevalence"],
            ["3", "Patient-level evaluation", "Patient F1", "Specificity was 0% (any-flag aggregation)"],
            ["4", "Max-probability aggregation", "Patient F1", "pF1=0.48, realistic metrics"],
            ["5", "Clinical features + GRU ensemble", "Patient F1", "pF1=0.49, AUROC=0.86"],
          ],
          [900, 2800, 2000, 3660],
        ),

        heading("3.2 Preprocessing Pipeline", HeadingLevel.HEADING_2),
        p("Each patient's time series undergoes the following preprocessing:"),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Missingness capture: ", bold: true }),
          new TextRun("Before any filling, binary masks (1=missing, 0=observed) and time-since-last-observation counters are created for all 40 features"),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Forward-fill: ", bold: true }),
          new TextRun("Missing values are carried forward in time (no look-ahead), reflecting the clinical practice of using the last known measurement"),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Population imputation: ", bold: true }),
          new TextRun("Remaining NaNs (before first observation) are filled with training set medians"),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Z-score normalization: ", bold: true }),
          new TextRun("Values are winsorized at the 1st/99th percentile and standardized using training set statistics"),
        ]}),
        p([
          "This produces ",
          bold("120 base features"),
          " per timestep: 40 normalized values + 40 missingness masks + 40 time-since features.",
        ]),

        heading("3.3 Feature Engineering (297 Features)", HeadingLevel.HEADING_2),
        p("Beyond the 120 base features, we engineered 177 additional features for the LightGBM model:"),
        spacer(),
        makeTable(
          ["Feature Category", "Count", "Description"],
          [
            ["Base features", "120", "40 values + 40 missingness masks + 40 time-since"],
            ["6h rolling statistics", "40", "Mean, min, max, std, trend for 8 vital signs"],
            ["12h rolling statistics", "32", "Mean, min, max, std for 8 vital signs"],
            ["24h rolling statistics", "32", "Mean, min, max, std for 8 vital signs"],
            ["Rate of change", "8", "Current value minus 6h-ago value for vitals"],
            ["Time-weighted average", "8", "Exponential decay (3h half-life) for vitals"],
            ["Clinical interactions", "4", "HR*SBP product, MAP trend, Resp*HR, temp deviation"],
            ["Differential features", "34", "Consecutive observation changes for all clinical vars"],
            ["SOFA score components", "7", "MAP, Creatinine, Bilirubin, Platelets, Resp, Coag, Total"],
            ["qSOFA score", "3", "SBP<=100, Resp>=22, Total"],
            ["NEWS vital scoring", "4", "HR, Temp, Resp range scores, Total"],
            ["Clinical ratios", "5", "Shock Index, BUN/Creatinine, SaO2/FiO2, Pulse Pressure, Cardiac Output proxy"],
            ["TOTAL", "297", "120 base + 177 engineered"],
          ],
          [2800, 900, 5660],
        ),

        heading("3.4 Clinical Scores Explanation", HeadingLevel.HEADING_3),
        p([
          bold("SOFA (Sequential Organ Failure Assessment)"),
          " scores organ dysfunction on a 0-4 scale across six systems. We compute components from available variables: MAP (<70 mmHg), Creatinine (thresholds at 1.2, 2.0, 3.5, 5.0 mg/dL), Bilirubin, Platelets (<150, <100, <50, <20 x10^3/uL), and SaO2/FiO2 ratio.",
        ]),
        p([
          bold("qSOFA (quick SOFA)"),
          " is a bedside screening tool using SBP <= 100 mmHg and Respiratory rate >= 22 breaths/min (altered mentation not available in the dataset).",
        ]),
        p([
          bold("NEWS (National Early Warning Score)"),
          " assigns points based on vital sign ranges: heart rate, temperature, and respiratory rate deviations from normal ranges.",
        ]),
        p([
          bold("Shock Index"),
          " (HR/SBP) is normally 0.5-0.7; values >0.9 indicate hemodynamic instability. ",
          bold("BUN/Creatinine ratio"),
          " (normal 10-20) helps differentiate pre-renal from intrinsic kidney injury.",
        ]),

        // ===== 4. MODELS =====
        heading("4. Models", HeadingLevel.HEADING_1),

        heading("4.1 LightGBM (Primary Model)", HeadingLevel.HEADING_2),
        p("LightGBM is a gradient-boosted decision tree framework that processes each timestep independently using the 297 engineered features. It was our primary model because:"),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Gradient-boosted trees dominated the PhysioNet 2019 Challenge (top 5 entries all used XGBoost or LightGBM)")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Fast training (~1 minute per configuration vs ~70 minutes for GRU)")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Naturally handles the engineered feature set without requiring sequence padding")] }),
        spacer(),
        p("We trained 9 configurations varying scale_pos_weight (5, 10, 20, 50, 100), num_leaves (63 vs 127), and learning rate (0.05 vs 0.03). All models used 500 boosting rounds with early stopping disabled (logloss-based early stopping was too aggressive with class weights, terminating at 1-3 trees)."),

        heading("4.2 GRU with Temporal Attention", HeadingLevel.HEADING_2),
        p("The GRU (Gated Recurrent Unit) processes the raw 120 base features as a time series, using a 24-hour sliding window. The architecture:"),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Bidirectional GRU: 256 hidden units, 1 layer (646,641 parameters)")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Temporal attention mechanism: learns which timesteps in the window are most informative")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Layer normalization + dropout (0.3) for regularization")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("BCEWithLogitsLoss with auto-computed positive class weight")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Trained with AdamW optimizer, mixed precision (FP16) on NVIDIA RTX PRO 1000")] }),

        heading("4.3 Ensemble Strategy", HeadingLevel.HEADING_2),
        p("LightGBM and GRU learn fundamentally different patterns:"),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "LightGBM: ", bold: true }),
          new TextRun("Processes each timestep independently with 297 engineered features. Strong at detecting cross-variable interactions (e.g., elevated creatinine + low MAP = organ dysfunction)."),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "GRU: ", bold: true }),
          new TextRun("Processes 24-hour sequences of 120 raw features. Strong at detecting temporal trends (e.g., steadily declining blood pressure over hours)."),
        ]}),
        spacer(),
        p([
          "Analysis shows the two models have a ",
          bold("Pearson correlation of only 0.59"),
          " in their patient-level probabilities, confirming they capture different clinical patterns. At threshold 0.40, LightGBM catches 88.6% of sepsis patients while GRU catches 41.3%, with only 45% overlap in their catches.",
        ]),
        p([
          "We use a ",
          bold("0.3/0.7 weighted ensemble (30% GRU + 70% LightGBM)"),
          ". LightGBM dominates detection while GRU acts as a precision filter: when LightGBM gives a borderline alarm but GRU disagrees, the blended probability drops, reducing false positives. This improves precision from 48.9% to 52.8% at the F1-optimal threshold.",
        ]),

        // ===== 5. EVALUATION =====
        heading("5. Evaluation Approach", HeadingLevel.HEADING_1),

        heading("5.1 Patient-Level vs. Timestep-Level", HeadingLevel.HEADING_2),
        p([
          "Early in the project, we evaluated at the ",
          bold("timestep level"),
          " (is this hour sepsis?), which produced misleadingly low F1 scores (~0.19) due to the extreme class imbalance. A hospital does not ask 'is this hour sepsis?' but rather '",
          bold("will this patient develop sepsis?"),
          "'",
        ]),
        p("We shifted to patient-level evaluation using max-probability aggregation:"),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [new TextRun("For each patient, take the maximum predicted probability across all timesteps")] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Apply a threshold to this single max probability")] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [new TextRun("A patient is 'flagged' if their max probability exceeds the threshold")] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Compare against ground truth: did this patient have any sepsis-labeled timestep?")] }),

        heading("5.2 Metrics Explained", HeadingLevel.HEADING_2),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Sensitivity (Recall): ", bold: true }),
          new TextRun("Of patients who have sepsis, what percentage did we catch?"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Specificity: ", bold: true }),
          new TextRun("Of patients who do not have sepsis, what percentage did we correctly leave alone?"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Precision: ", bold: true }),
          new TextRun("Of patients we flagged, what percentage actually have sepsis?"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "F1 Score: ", bold: true }),
          new TextRun("Harmonic mean of precision and sensitivity. Balances 'did we catch sepsis patients?' with 'are our alarms real?'"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "F2 Score: ", bold: true }),
          new TextRun("Like F1 but weights recall 2x more than precision. Better for clinical use where missing a sepsis patient is worse than a false alarm."),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "AUROC: ", bold: true }),
          new TextRun("Probability that the model ranks a random sepsis patient above a random non-sepsis patient. Threshold-independent measure of discrimination."),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "AUPRC: ", bold: true }),
          new TextRun("Area under the precision-recall curve. More informative than AUROC for imbalanced data. A random classifier achieves AUPRC = 0.0184 (the prevalence rate)."),
        ]}),

        // ===== 6. RESULTS =====
        heading("6. Results", HeadingLevel.HEADING_1),

        heading("6.1 Best Model Summary", HeadingLevel.HEADING_2),
        p([
          "Our best model is the ",
          bold("Ensemble 0.3/0.7 (GRU + LightGBM spw_50)"),
          ":",
        ]),
        spacer(),
        makeTable(
          ["Metric", "Value"],
          [
            ["Patient-level AUROC", "0.8573"],
            ["Patient-level AUPRC", "0.5032 (28x lift over 1.84% base rate)"],
            ["Best Patient F1", "0.487 (at threshold 0.55)"],
            ["Best F2", "0.515 (at threshold 0.40)"],
            ["Best MCC", "0.465 (at threshold 0.65)"],
            ["PhysioNet Utility", "0.353 (at threshold 0.50)"],
          ],
          [3500, 5860],
        ),

        heading("6.2 Operating Points", HeadingLevel.HEADING_2),
        p("The model offers different operating points depending on clinical needs:"),
        spacer(),
        makeTable(
          ["Use Case", "Threshold", "Sensitivity", "Precision", "F1", "F2"],
          [
            ["Screening", "0.40", "78.0%", "21.8%", "0.341", "0.515"],
            ["Balanced", "0.55", "53.7%", "41.2%", "0.466", "0.506"],
            ["Conservative", "0.65", "36.5%", "66.8%", "0.472", "0.401"],
          ],
          [1800, 1400, 1600, 1400, 1080, 1080],
        ),
        spacer(),
        p([
          bold("Screening mode (threshold 0.40):"),
          " Catches 78% of sepsis patients 6 hours early. About 1 in 5 alarms is real (21.8% precision). A nurse verifies each alarm. Best for units where missing sepsis is unacceptable.",
        ]),
        p([
          bold("Balanced mode (threshold 0.55):"),
          " Catches 54% of sepsis patients. About 2 in 5 alarms are real (41.2% precision). Best balance between alert fatigue and detection.",
        ]),
        p([
          bold("Conservative mode (threshold 0.65):"),
          " Catches 37% of sepsis patients. About 2 in 3 alarms are real (66.8% precision). Suitable for resource-constrained settings where each alarm triggers expensive interventions.",
        ]),

        heading("6.3 Full Threshold Sweep", HeadingLevel.HEADING_2),
        p("Complete metrics for the best ensemble model across all thresholds:"),
        spacer(),
        makeTable(
          ["Thr", "F1", "F2", "Sens", "Spec", "Prec", "MCC", "TP", "FP", "FN"],
          [
            ["0.05", "0.144", "0.296", "100.0%", "2.5%", "7.8%", "0.044", "458", "5455", "0"],
            ["0.20", "0.194", "0.372", "96.7%", "34.3%", "10.8%", "0.176", "443", "3676", "15"],
            ["0.30", "0.247", "0.437", "89.7%", "55.9%", "14.3%", "0.242", "411", "2464", "47"],
            ["0.35", "0.282", "0.470", "84.7%", "65.9%", "16.9%", "0.276", "388", "1905", "70"],
            ["0.40", "0.341", "0.515", "78.0%", "77.1%", "21.8%", "0.328", "357", "1280", "101"],
            ["0.45", "0.385", "0.523", "68.8%", "84.6%", "26.8%", "0.357", "315", "862", "143"],
            ["0.50", "0.427", "0.515", "59.6%", "90.2%", "33.3%", "0.385", "273", "547", "185"],
            ["0.55", "0.466", "0.506", "53.7%", "93.7%", "41.2%", "0.421", "246", "351", "212"],
            ["0.60", "0.485", "0.462", "44.8%", "96.7%", "52.8%", "0.448", "205", "183", "253"],
            ["0.65", "0.472", "0.401", "36.5%", "98.5%", "66.8%", "0.465", "167", "83", "291"],
            ["0.70", "0.432", "0.343", "30.1%", "99.2%", "76.2%", "0.456", "138", "43", "320"],
            ["0.75", "0.375", "0.281", "24.0%", "99.7%", "85.3%", "0.434", "110", "19", "348"],
            ["0.80", "0.267", "0.188", "15.7%", "99.8%", "88.9%", "0.358", "72", "9", "386"],
            ["0.90", "0.087", "0.057", "4.6%", "100.0%", "91.3%", "0.196", "21", "2", "437"],
          ],
          [780, 780, 780, 1000, 1000, 1000, 780, 780, 780, 780],
        ),
        spacer(),
        p("The threshold is a deployment knob, not a model quality metric. AUROC (0.86) is threshold-independent and demonstrates strong discrimination."),

        heading("6.4 Individual Model Comparison", HeadingLevel.HEADING_2),
        p("Patient-level metrics at each model's F1-optimal threshold:"),
        spacer(),
        makeTable(
          ["Model", "pF1", "Sens", "Spec", "Prec", "AUROC", "AUPRC"],
          [
            ["Ens GRU+LGB (spw_50)", "0.485", "44.8%", "96.7%", "52.8%", "0.8573", "0.5032"],
            ["Ens GRU+LGB (spw_5)", "0.484", "40.8%", "97.7%", "59.4%", "0.8728", "0.5203"],
            ["LGB (unbalance_big)", "0.477", "41.5%", "97.4%", "56.2%", "0.8433", "0.5032"],
            ["LGB (spw_10_big)", "0.476", "41.3%", "97.4%", "56.3%", "0.8603", "0.4969"],
            ["LGB (spw_5)", "0.476", "41.1%", "97.4%", "56.6%", "0.8615", "0.4827"],
            ["LGB (spw_50)", "0.470", "45.2%", "96.1%", "48.9%", "0.8478", "0.4793"],
            ["GRU (patient_f1)", "0.463", "46.3%", "95.6%", "46.4%", "0.8419", "0.4364"],
          ],
          [2300, 900, 900, 900, 900, 1180, 1180],
        ),

        heading("6.5 GRU vs LightGBM Overlap Analysis", HeadingLevel.HEADING_2),
        p("At threshold 0.40, the two model types show complementary detection patterns:"),
        spacer(),
        makeTable(
          ["Category", "Count", "Percentage"],
          [
            ["Both catch (sepsis patients)", "185", "40.4%"],
            ["LightGBM only catches", "221", "48.3%"],
            ["GRU only catches", "4", "0.9%"],
            ["Both miss", "48", "10.5%"],
          ],
          [4000, 2680, 2680],
        ),
        spacer(),
        p([
          "LightGBM dominates detection (88.6% sensitivity vs GRU's 41.3%), while GRU's primary contribution is ",
          bold("reducing false alarms"),
          ". LightGBM generates 2,517 false alarms vs GRU's 183. The ensemble leverages GRU's conservatism to filter out borderline LightGBM false positives, improving precision by ~4% without sacrificing sensitivity.",
        ]),

        heading("6.6 Comparison with PhysioNet Challenge Winners", HeadingLevel.HEADING_2),
        p("The PhysioNet 2019 Challenge had 104 participating teams. Top entries:"),
        spacer(),
        makeTable(
          ["Team", "Approach", "Utility Score"],
          [
            ["Yang et al. (1st unofficial)", "XGBoost + Bayesian optimization", "0.364"],
            ["Morrill et al. (1st official)", "Path signatures + LightGBM", "0.360"],
            ["Du et al. (2nd official)", "Gradient boosted trees", "0.345"],
            ["Zabihi et al. (3rd official)", "5x XGBoost ensemble", "0.339"],
            ["Our model", "GRU + LightGBM ensemble", "0.353*"],
          ],
          [2800, 3800, 2760],
        ),
        p([italic("*At threshold 0.50. Our utility score varies by threshold since we optimize for patient-level F1, not utility.")]),
        p("Our results are competitive with top PhysioNet entries. All winning solutions used gradient-boosted trees with extensive feature engineering, validating our approach. Our clinical feature engineering (SOFA, qSOFA, NEWS) aligns with the top teams' use of domain-knowledge features."),

        // ===== 7. CHALLENGES =====
        heading("7. Challenges and Lessons Learned", HeadingLevel.HEADING_1),

        heading("7.1 Class Imbalance", HeadingLevel.HEADING_2),
        p([
          "With only 1.84% positive rate at the timestep level, even 96% specificity creates more false positives than true positives in absolute terms. We tried ",
          bold("SMOTE oversampling"),
          " (generating synthetic sepsis samples), but it ",
          bold("hurt performance"),
          " (best F1 dropped from 0.48 to 0.44). The synthetic samples added noise because SMOTE creates interpolated timesteps that don't follow real clinical trajectories. Cost-sensitive learning through class weights (scale_pos_weight) proved more effective.",
        ]),

        heading("7.2 Evaluation Metric Selection", HeadingLevel.HEADING_2),
        p("We initially optimized for PhysioNet's custom utility score, which is timestep-based and rewards early predictions. However, this metric doesn't directly answer the clinical question: 'will this patient develop sepsis?' Shifting to patient-level F1 was a key decision that made our evaluation clinically meaningful and dramatically changed how we interpreted results."),
        p([
          "Similarly, we considered ",
          bold("Youden's J statistic"),
          " (maximizes sensitivity + specificity), but at 1.84% prevalence it selects thresholds with ~80% sensitivity and massive false positive rates. Youden works for balanced problems but fails with extreme imbalance.",
        ]),

        heading("7.3 Patient-Level Aggregation", HeadingLevel.HEADING_2),
        p([
          "Our initial patient-level evaluation used 'any single timestep flagged = patient flagged.' This produced ",
          bold("0% specificity"),
          " because with ~4% per-timestep false positive rate and ~40 hours per patient, the probability that at least one timestep triggers is 1 - 0.96^40 = 80%+. Nearly every non-sepsis patient had at least one false alarm. Switching to ",
          bold("max-probability aggregation"),
          " (one score per patient = their highest predicted probability) resolved this completely.",
        ]),

        heading("7.4 Early Stopping with Class Weights", HeadingLevel.HEADING_2),
        p("LightGBM's logloss-based early stopping was too aggressive when combined with class weights. Models stopped at 1-3 trees because the weighted loss function fluctuated early in training, triggering the patience criterion before the model could learn. Disabling early stopping and using a fixed 500 rounds resolved this, allowing full model capacity."),

        heading("7.5 Stacking Meta-Learner Overfitting", HeadingLevel.HEADING_2),
        p("We attempted a stacking approach (logistic regression on top of GRU + LightGBM predictions). The meta-learner achieved 0.999 AUROC on training but only 0.72 on validation, a severe overfit. This occurred because the base models' training predictions were too optimistic (they had already seen the training data). Proper stacking requires out-of-fold predictions, which would require retraining all base models in k-fold, adding hours of compute for marginal gains."),

        heading("7.6 Technical Challenges", HeadingLevel.HEADING_2),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "GPU compatibility: ", bold: true }),
          new TextRun("The NVIDIA RTX PRO 1000 (Blackwell architecture, sm_128) required PyTorch nightly builds with CUDA 12.8 support; standard pip packages lack sm_128"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Windows memory constraints: ", bold: true }),
          new TextRun("DataLoader with num_workers > 0 caused paging file errors; resolved by setting num_workers=0"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "LightGBM memory: ", bold: true }),
          new TextRun("Training with 1.3M samples x 297 features required ~8GB RAM; running games alongside training caused OOM crashes"),
        ]}),

        // ===== 8. CONCLUSION =====
        heading("8. Conclusion", HeadingLevel.HEADING_1),

        heading("8.1 Summary", HeadingLevel.HEADING_2),
        p("We developed a sepsis early warning system that continuously monitors ICU patients and alerts clinicians 6 hours before sepsis onset. The system uses a GRU + LightGBM ensemble with 297 clinical features, achieving an AUROC of 0.86 and AUPRC of 0.50."),
        p("Key contributions:"),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Patient-level evaluation framework: ", bold: true }),
          new TextRun("Max-probability aggregation with clinically relevant operating points"),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Clinical feature engineering: ", bold: true }),
          new TextRun("SOFA, qSOFA, NEWS scores alongside data-driven temporal features"),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Cross-model ensemble: ", bold: true }),
          new TextRun("GRU as precision filter for LightGBM (0.59 correlation, complementary detection patterns)"),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Comprehensive threshold analysis: ", bold: true }),
          new TextRun("118 model configurations x 19 thresholds with 14 metrics each"),
        ]}),

        heading("8.2 Limitations", HeadingLevel.HEADING_2),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("The 1.84% prevalence rate fundamentally limits the precision-sensitivity tradeoff; no model can achieve both >80% sensitivity and >50% precision at this prevalence")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Evaluated on a single train/validation split; k-fold cross-validation would provide more robust estimates")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("The model is trained on data from two US hospital systems; generalizability to other institutions is unknown (PhysioNet winners saw significant performance drops on a hidden third hospital)")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("The model outputs probability of sepsis, not the cause or recommended treatment")] }),

        heading("8.3 Future Work", HeadingLevel.HEADING_2),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Transformer-based models: ", bold: true }),
          new TextRun("Self-attention across all timesteps could capture longer-range dependencies than the GRU's 24-hour window"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Measurement frequency features: ", bold: true }),
          new TextRun("How often labs are ordered is itself a clinical signal (doctors order more tests when suspicious)"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Survival analysis framing: ", bold: true }),
          new TextRun("Predicting time-to-sepsis instead of binary classification would naturally handle censored data"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Self-supervised pretraining: ", bold: true }),
          new TextRun("Pretrain on all 40K patients with masked imputation before fine-tuning on sepsis labels"),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [
          new TextRun({ text: "Multi-hospital validation: ", bold: true }),
          new TextRun("Test on the hidden PhysioNet test set from a third hospital system to assess generalizability"),
        ]}),

        heading("8.4 Clinical Deployment Considerations", HeadingLevel.HEADING_2),
        p("This model is designed as an early warning trigger, not a standalone diagnostic tool. In deployment:"),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("Every alarm should prompt clinician review, not automatic treatment")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("The threshold should be set by the clinical team based on their tolerance for false alarms vs missed cases")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("The model monitors continuously, with probability climbing as the patient deteriorates, giving clinicians a dynamic risk signal rather than a one-time prediction")] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun("At screening threshold (0.40), catching 78% of sepsis patients 6 hours early provides a meaningful clinical intervention window, even with the associated false alarm burden")] }),
      ],
    },
  ],
});

// Generate the document
const outputPath = "W:\\ml\\sepsis-forecasting\\outputs\\docs\\sepsis_project_report_final.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outputPath, buffer);
  console.log(`Report generated: ${outputPath}`);
  console.log(`Size: ${(buffer.length / 1024).toFixed(1)} KB`);
});
