/**
 * Handwritten Digit Recognition Client Studio (DigitVision AI).
 * Handles HTML5 Canvas drawing, drag-and-drop file uploads, AJAX predictions,
 * dynamic probability bar visualizations, Grad-CAM toggle, and prediction history tracking.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const canvas = document.getElementById("digit-canvas");
    const ctx = canvas.getContext("2d");
    const brushSizeInput = document.getElementById("brush-size");
    const brushSizeVal = document.getElementById("brush-size-val");
    const btnClear = document.getElementById("btn-clear");
    const btnPredictCanvas = document.getElementById("btn-predict-canvas");

    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const uploadPreviewContainer = document.getElementById("upload-preview-container");
    const uploadPreviewImg = document.getElementById("upload-preview-img");
    const btnPredictUpload = document.getElementById("btn-predict-upload");

    const predictedDigitEl = document.getElementById("predicted-digit");
    const confidenceValEl = document.getElementById("confidence-val");
    const confidenceFillEl = document.getElementById("confidence-fill");
    const latencyValEl = document.getElementById("latency-val");
    const probBarsContainer = document.getElementById("prob-bars");

    const gradcamToggle = document.getElementById("gradcam-toggle");
    const gradcamImg = document.getElementById("gradcam-img");
    const gradcamPlaceholder = document.getElementById("gradcam-placeholder");

    const btnExportReport = document.getElementById("btn-export-report");
    const historyTbody = document.getElementById("history-tbody");
    const btnModelInfo = document.getElementById("btn-model-info");
    const modelModal = document.getElementById("model-modal");
    const modalClose = document.getElementById("modal-close");

    // State Variables
    let isDrawing = false;
    let brushSize = parseInt(brushSizeInput.value, 10);
    let lastPredictionData = null;

    // Initialize Canvas Background (Pure Black for MNIST)
    function initCanvas() {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = brushSize;
    }
    initCanvas();

    // Brush Size Slider Event
    brushSizeInput.addEventListener("input", (e) => {
        brushSize = parseInt(e.target.value, 10);
        brushSizeVal.textContent = `${brushSize}px`;
        ctx.lineWidth = brushSize;
    });

    // Clear Canvas
    btnClear.addEventListener("click", () => {
        initCanvas();
        resetPredictionDisplay();
    });

    // Mouse & Touch Drawing Mechanics
    function startDrawing(e) {
        isDrawing = true;
        draw(e);
    }

    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath();
    }

    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: (clientX - rect.left) * (canvas.width / rect.width),
            y: (clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const coords = getCanvasCoords(e);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
    }

    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);

    canvas.addEventListener("touchstart", startDrawing);
    canvas.addEventListener("touchmove", draw);
    canvas.addEventListener("touchend", stopDrawing);

    // Tab Switching Logic
    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            tabBtns.forEach((b) => b.classList.remove("active"));
            tabContents.forEach((c) => c.classList.remove("active"));
            btn.classList.add("active");
            const targetTab = btn.getAttribute("data-tab");
            document.getElementById(targetTab).classList.add("active");
        });
    });

    // Preset Digits Generator
    document.querySelectorAll(".preset-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const digit = btn.getAttribute("data-digit");
            drawPresetDigit(digit);
        });
    });

    function drawPresetDigit(digitStr) {
        initCanvas();
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 180px Outfit, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(digitStr, canvas.width / 2, canvas.height / 2 + 10);
    }

    // Drag and Drop Upload Handlers
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file.");
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreviewImg.src = e.target.result;
            uploadPreviewContainer.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }

    // Prediction Triggers
    btnPredictCanvas.addEventListener("click", () => {
        const base64Data = canvas.toDataURL("image/png");
        sendPredictionRequest({ image: base64Data });
    });

    btnPredictUpload.addEventListener("click", () => {
        if (uploadPreviewImg.src) {
            sendPredictionRequest({ image: uploadPreviewImg.src });
        }
    });

    // AJAX Prediction API Call
    async function sendPredictionRequest(payload) {
        setLoadingState(true);
        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Prediction error: ${response.statusText}`);
            }

            const data = await response.json();
            lastPredictionData = data;
            renderPredictionResults(data);
            fetchPredictionHistory();
        } catch (error) {
            console.error("Prediction failed:", error);
            alert("Error processing prediction. Check console for details.");
        } finally {
            setLoadingState(false);
        }
    }

    // Render Prediction Output to Dashboard
    function renderPredictionResults(data) {
        predictedDigitEl.textContent = data.predicted_digit;
        confidenceValEl.textContent = `${data.confidence_percentage}%`;
        confidenceFillEl.style.width = `${data.confidence_percentage}%`;
        latencyValEl.innerHTML = `<i class="fa-solid fa-bolt"></i> Latency: ${data.processing_time_ms} ms`;

        // Render Probabilities Horizontal Bars
        probBarsContainer.innerHTML = "";
        data.probabilities.forEach((prob, digit) => {
            const pct = (prob * 100).toFixed(1);
            const isTop = digit === data.predicted_digit;

            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML = `
                <span class="prob-digit">${digit}</span>
                <div class="prob-track">
                    <div class="prob-fill ${isTop ? 'top-pred' : ''}" style="width: ${pct}%"></div>
                </div>
                <span class="prob-percent">${pct}%</span>
            `;
            probBarsContainer.appendChild(row);
        });

        // Render Grad-CAM Heatmap
        if (data.gradcam_image && gradcamToggle.checked) {
            gradcamImg.src = data.gradcam_image;
            gradcamImg.classList.remove("hidden");
            gradcamPlaceholder.classList.add("hidden");
        }

        btnExportReport.disabled = false;
    }

    function resetPredictionDisplay() {
        predictedDigitEl.textContent = "?";
        confidenceValEl.textContent = "--%";
        confidenceFillEl.style.width = "0%";
        latencyValEl.innerHTML = `<i class="fa-solid fa-bolt"></i> Latency: -- ms`;
        probBarsContainer.innerHTML = `<div class="empty-state">Run a prediction to view probability spectrum</div>`;
        gradcamImg.classList.add("hidden");
        gradcamPlaceholder.classList.remove("hidden");
        btnExportReport.disabled = true;
    }

    function setLoadingState(isLoading) {
        if (isLoading) {
            predictedDigitEl.innerHTML = `<i class="fa-solid fa-spinner fa-spin" style="font-size: 36px;"></i>`;
        }
    }

    // Grad-CAM Toggle Switch
    gradcamToggle.addEventListener("change", (e) => {
        if (lastPredictionData && lastPredictionData.gradcam_image) {
            if (e.target.checked) {
                gradcamImg.src = lastPredictionData.gradcam_image;
                gradcamImg.classList.remove("hidden");
                gradcamPlaceholder.classList.add("hidden");
            } else {
                gradcamImg.classList.add("hidden");
                gradcamPlaceholder.classList.remove("hidden");
            }
        }
    });

    // Fetch and Populate History Log Table
    async function fetchPredictionHistory() {
        try {
            const res = await fetch("/api/history");
            const data = await res.json();
            historyTbody.innerHTML = "";

            if (!data.history || data.history.length === 0) {
                historyTbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted">No inferences recorded yet</td></tr>`;
                return;
            }

            data.history.forEach((item) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>#${item.id}</td>
                    <td><strong style="color: var(--accent-cyan); font-size: 16px;">${item.digit}</strong></td>
                    <td>${item.confidence}%</td>
                    <td>${item.latency} ms</td>
                    <td><span class="badge">Success</span></td>
                `;
                historyTbody.appendChild(tr);
            });
        } catch (err) {
            console.error("Failed to fetch history:", err);
        }
    }
    fetchPredictionHistory();

    // Export PDF Report Download Trigger
    btnExportReport.addEventListener("click", async () => {
        if (!lastPredictionData) return;
        try {
            const response = await fetch("/api/export-report", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(lastPredictionData)
            });

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "digit_prediction_report.pdf";
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (err) {
            console.error("Report download failed:", err);
        }
    });

    // Model Stats Modal Handlers
    btnModelInfo.addEventListener("click", async () => {
        modelModal.classList.remove("hidden");
        try {
            const res = await fetch("/api/model-info");
            const info = await res.json();

            const container = document.getElementById("modal-metrics-container");
            if (info.test_metrics && info.test_metrics.test_accuracy) {
                const accPct = (info.test_metrics.test_accuracy * 100).toFixed(2);
                const f1 = info.test_metrics.f1_score_weighted.toFixed(4);

                container.innerHTML = `
                    <div style="margin-top: 16px; padding: 12px; background: rgba(0,242,254,0.08); border-radius: 8px; border: 1px solid var(--accent-cyan);">
                        <h4 style="color: var(--accent-cyan); margin-bottom: 6px;">Evaluation Metrics (Test Set)</h4>
                        <p><strong>Accuracy:</strong> ${accPct}% | <strong>Weighted F1-Score:</strong> ${f1}</p>
                        <p><strong>Test Loss:</strong> ${info.test_metrics.test_loss.toFixed(4)}</p>
                    </div>
                `;
            } else {
                container.innerHTML = `<p style="margin-top: 12px; color: var(--text-muted);">Run evaluation script (python src/evaluate.py) to generate metrics.</p>`;
            }
        } catch (err) {
            console.error(err);
        }
    });

    modalClose.addEventListener("click", () => modelModal.classList.add("hidden"));
    window.addEventListener("click", (e) => {
        if (e.target === modelModal) modelModal.classList.add("hidden");
    });
});
