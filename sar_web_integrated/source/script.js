// ============================================================
// SAR SHIP DETECTOR - FRONTEND SCRIPT
// ============================================================


// ============================================================
// ELEMENTS
// ============================================================

const fileInput = document.getElementById("file");
const dropArea = document.getElementById("drop");
const runButton = document.getElementById("run");
const statusBox = document.getElementById("status");


// ============================================================
// CANVAS ELEMENTS
// ============================================================

const originalCanvas = document.getElementById("original");
const processedCanvas = document.getElementById("processed");
const cfarCanvas = document.getElementById("cfar");
const yoloCanvas = document.getElementById("yolo");
const compareCanvas = document.getElementById("compare");


// ============================================================
// DISPLAY IMAGE ON CANVAS
// ============================================================

function displayImage(canvas, value) {

    if (!canvas || !value) {
        return;
    }


    /*
     * Backend may return:
     *
     * 1. A normal URL:
     *    "/results/web/image.png"
     *
     * 2. An object:
     *    {
     *        url: "/results/web/image.png"
     *    }
     */


    let url = value;


    if (typeof value === "object") {

        url =
            value.url ||
            value.image ||
            value.path ||
            value.src ||
            value.visualization ||
            value.file ||
            "";

    }


    if (!url || typeof url !== "string") {

        console.warn(
            "Invalid image URL:",
            value
        );

        return;
    }


    const ctx =
        canvas.getContext("2d");


    const img =
        new Image();


    img.onload = function () {

        canvas.width =
            img.width;

        canvas.height =
            img.height;


        ctx.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );


        ctx.drawImage(
            img,
            0,
            0
        );

    };


    img.onerror = function () {

        console.error(
            "Could not load image:",
            url
        );

    };


    /*
     * Cache busting.
     * Prevents browser from displaying an old result.
     */

    const separator =
        url.includes("?")
            ? "&"
            : "?";


    img.src =
        url +
        separator +
        "t=" +
        Date.now();

}


// ============================================================
// EXTRACT CFAR IMAGE URL
// ============================================================

function getCFARImage(cfar) {

    if (!cfar) {
        return null;
    }


    /*
     * Backend may return CFAR directly as a URL.
     */

    if (typeof cfar === "string") {
        return cfar;
    }


    /*
     * Backend may return CFAR as an object.
     */

    if (typeof cfar === "object") {

        return (
            cfar.url ||
            cfar.image ||
            cfar.path ||
            cfar.src ||
            cfar.visualization ||
            cfar.file ||
            null
        );

    }


    return null;

}


// ============================================================
// FILE SELECTION
// ============================================================

fileInput.addEventListener(
    "change",
    function () {

        if (!fileInput.files.length) {
            return;
        }


        const file =
            fileInput.files[0];


        statusBox.textContent =
            "Image selected: " +
            file.name +
            " — ready for detection.";

    }
);


// ============================================================
// DRAG & DROP
// ============================================================

dropArea.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();


        dropArea.style.borderColor =
            "#28e57b";

    }
);


dropArea.addEventListener(
    "dragleave",
    function () {

        dropArea.style.borderColor =
            "#376075";

    }
);


dropArea.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();


        dropArea.style.borderColor =
            "#376075";


        const files =
            event.dataTransfer.files;


        if (!files.length) {
            return;
        }


        fileInput.files =
            files;


        statusBox.textContent =
            "Image selected: " +
            files[0].name +
            " — ready for detection.";

    }
);


// ============================================================
// UPDATE DETECTION DETAILS TABLE
// ============================================================

function updateDetectionDetails(detections) {

    const details =
        document.getElementById("details");


    if (!details) {
        return;
    }


    details.innerHTML = "";


    if (
        !detections ||
        !Array.isArray(detections) ||
        detections.length === 0
    ) {

        details.innerHTML = `
            <tr>
                <td colspan="4">
                    No detection yet.
                </td>
            </tr>
        `;

        return;
    }


    detections.forEach(
        function (detection, index) {

            const row =
                document.createElement("tr");


            const bbox =
                detection.bbox ||
                [0, 0, 0, 0];


            const confidence =
                Number(
                    detection.confidence || 0
                );


            row.innerHTML = `
                <td>
                    ${detection.id || index + 1}
                </td>

                <td>
                    ${detection.label || "ship"}
                </td>

                <td>
                    ${confidence.toFixed(3)}
                </td>

                <td>
                    (${bbox[0]},
                     ${bbox[1]},
                     ${bbox[2]},
                     ${bbox[3]})
                </td>
            `;


            details.appendChild(row);

        }
    );

}


// ============================================================
// UPDATE CURRENT IMAGE METRICS
// ============================================================

function updateCurrentMetrics(data) {

    const cfar =
        data.cfar || {};


    const totalDetections =
        Number(
            data.total_detections || 0
        );


    const averageConfidence =
        Number(
            data.average_confidence || 0
        );


    /*
     * Support multiple possible CFAR property names.
     */

    const caDetections = Number(cfar.ca_candidates ?? 0);
    const osDetections = Number(cfar.os_candidates ?? cfar.detections ?? 0);
    const cfarDetections = osDetections;


    const osMetrics = cfar.os_metrics || {};
    const cfarTP = Number(osMetrics.tp ?? cfar.tp ?? cfar.TP ?? 0);
    const cfarFP = Number(osMetrics.fp ?? cfar.fp ?? cfar.FP ?? cfar.false_positives ?? 0);
    const cfarFN = Number(osMetrics.fn ?? cfar.fn ?? cfar.FN ?? cfar.false_negatives ?? 0);


    // ========================================================
    // TOP CURRENT-IMAGE METRIC CARDS
    // ========================================================

    const currentShips =
        document.getElementById(
            "currentShips"
        );


    const currentConfidence =
        document.getElementById(
            "currentConfidence"
        );


    const currentCFAR =
        document.getElementById(
            "currentCFAR"
        );


    const currentFP =
        document.getElementById(
            "currentFP"
        );


    if (currentShips) {

        currentShips.textContent =
            totalDetections;

    }


    if (currentConfidence) {

        currentConfidence.textContent =
            averageConfidence.toFixed(3);

    }


    if (currentCFAR) {

        currentCFAR.textContent =
            cfarDetections;

    }


    if (currentFP) {

        currentFP.textContent =
            cfarFP;

    }

    const cfarDetectionCount = document.getElementById("cfarDetectionCount");
    const osCfarDetectionCount = document.getElementById("osCfarDetectionCount");

    if (cfarDetectionCount) {
        cfarDetectionCount.textContent = caDetections;
    }

    if (osCfarDetectionCount) {
        osCfarDetectionCount.textContent = osDetections;
    }


    // ========================================================
    // CURRENT IMAGE TABLE
    // ========================================================

    const currentShipsTable =
        document.getElementById(
            "currentShipsTable"
        );


    const currentConfidenceTable =
        document.getElementById(
            "currentConfidenceTable"
        );


    const currentCFARTable =
        document.getElementById(
            "currentCFARTable"
        );

    const currentCACFARTable = document.getElementById("currentCACFARTable");
    const currentOSCFARTable = document.getElementById("currentOSCFARTable");


    const currentTPTable =
        document.getElementById(
            "currentTPTable"
        );


    const currentFPTable =
        document.getElementById(
            "currentFPTable"
        );


    const currentFNTable =
        document.getElementById(
            "currentFNTable"
        );


    if (currentShipsTable) {

        currentShipsTable.textContent =
            totalDetections;

    }


    if (currentConfidenceTable) {

        currentConfidenceTable.textContent =
            averageConfidence.toFixed(3);

    }


    if (currentCFARTable) {

        currentCFARTable.textContent =
            cfarDetections;

    }

    if (currentCACFARTable) {
        currentCACFARTable.textContent = caDetections;
    }

    if (currentOSCFARTable) {
        currentOSCFARTable.textContent = osDetections;
    }


    if (currentTPTable) {

        currentTPTable.textContent =
            cfarTP;

    }


    if (currentFPTable) {

        currentFPTable.textContent =
            cfarFP;

    }


    if (currentFNTable) {

        currentFNTable.textContent =
            cfarFN;

    }

}


// ============================================================
// UPDATE YOLO MODEL PERFORMANCE METRICS
// ============================================================

function updateModelMetrics(data) {

    if (!data || !data.model_metrics) {
        return;
    }

    const metrics = data.model_metrics;

    const precision = Number(metrics.precision ?? 0);
    const recall = Number(metrics.recall ?? 0);
    const map50 = Number(metrics.map50 ?? 0);
    const map5095 = Number(metrics.map5095 ?? 0);
    const commonF1 = Number(metrics.common_f1 ?? 0);

    // Official held-out test metrics.
    const metricCards = document.querySelectorAll('.metrics .metric');
    const values = [precision, recall, map50, map5095];

    values.forEach(function(value, index) {
        const card = metricCards[index];
        if (!card) return;
        const valueElement = card.querySelector('b');
        if (valueElement) valueElement.textContent = value.toFixed(3);
    });

    // Performance table.
    const setText = function(id, value) {
        const element = document.getElementById(id);
        if (element) element.textContent = value.toFixed(3);
    };

    setText('modelPrecisionTable', precision);
    setText('modelRecallTable', recall);
    setText('modelF1Table', commonF1);
    setText('modelMap50Table', map50);
    setText('modelMap5095Table', map5095);
}


// ============================================================
// UPDATE SUMMARY
// ============================================================

function updateSummary(data) {

    const summary =
        document.querySelector(
            ".summary"
        );


    if (!summary) {
        return;
    }


    const total =
        Number(
            data.total_detections || 0
        );


    const confidence =
        Number(
            data.average_confidence || 0
        );


    summary.innerHTML = `
        <span>
            Total Detections:
            ${total}
        </span>

        <span>
            Avg Confidence:
            ${confidence.toFixed(3)}
        </span>
    `;

}


// ============================================================
// RUN DETECTION
// ============================================================

runButton.addEventListener(
    "click",
    async function () {


        // ====================================================
        // CHECK FILE
        // ====================================================

        if (!fileInput.files.length) {

            statusBox.textContent =
                "Please select a SAR image first.";

            return;

        }


        const file =
            fileInput.files[0];


        const formData =
            new FormData();


        formData.append(
            "file",
            file
        );


        // ====================================================
        // UPDATE UI
        // ====================================================

        runButton.disabled =
            true;


        runButton.textContent =
            "Processing...";


        statusBox.textContent =
            "Running CA-CFAR → OS-CFAR → YOLO11n. Please wait...";


        try {


            // =================================================
            // SEND IMAGE TO FLASK
            // =================================================

            const response =
                await fetch(
                    "/detect",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            // =================================================
            // READ RESPONSE
            // =================================================

            const data =
                await response.json();


            console.log(
                "Detection response:",
                data
            );


            // =================================================
            // CHECK RESPONSE
            // =================================================

            if (
                !response.ok ||
                !data.success
            ) {

                throw new Error(
                    data.error ||
                    "Detection failed."
                );

            }


            // =================================================
            // DISPLAY RESULT IMAGES
            // =================================================

            displayImage(
                originalCanvas,
                data.original
            );


            displayImage(
                processedCanvas,
                data.processed
            );


            /*
             * CFAR is returned as an object by the backend.
             * Extract its actual image URL first.
             */

            const cfarImage =
                getCFARImage(
                    data.cfar
                );


            displayImage(
                cfarCanvas,
                cfarImage
            );


            displayImage(
                yoloCanvas,
                data.yolo
            );


            displayImage(
                compareCanvas,
                data.comparison
            );


            // =================================================
            // UPDATE DETECTION DETAILS
            // =================================================

            updateDetectionDetails(
                data.detections
            );


            // =================================================
            // UPDATE CURRENT IMAGE METRICS
            // =================================================

            updateCurrentMetrics(
                data
            );


            // =================================================
            // UPDATE YOLO MODEL METRICS
            // =================================================

            updateModelMetrics(
                data
            );


            // =================================================
            // UPDATE SUMMARY
            // =================================================

            updateSummary(
                data
            );


            // =================================================
            // SUCCESS MESSAGE
            // =================================================

            const total =
                Number(
                    data.total_detections || 0
                );


            statusBox.textContent =
                "Detection completed successfully. " +
                total +
                " ship(s) detected.";


        }


        catch (error) {

            console.error(
                "Detection error:",
                error
            );


            statusBox.textContent =
                "Error: " +
                error.message;

        }


        finally {

            runButton.disabled =
                false;


            runButton.textContent =
                "Run Detection ❯";

        }

    }
);