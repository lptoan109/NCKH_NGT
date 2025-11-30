// recorder.js - Bản Final (Đã bỏ hiển thị độ tin cậy)

import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

// --- HÀM TRỢ GIÚP: Chuyển Blob sang Base64 ---
function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

// --- HÀM TRỢ GIÚP: Kiểm tra file âm thanh im lặng ---
async function isAudioSilent(audioBlob) {
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const channelData = audioBuffer.getChannelData(0);
        
        let sumSquares = 0.0;
        for (let i = 0; i < channelData.length; i++) {
            sumSquares += channelData[i] * channelData[i];
        }
        const rms = Math.sqrt(sumSquares / channelData.length);
        const SILENCE_THRESHOLD = 0.01; 
        
        console.log("Audio RMS (âm lượng):", rms);
        return rms < SILENCE_THRESHOLD;
    } catch (error) {
        console.error("Lỗi phân tích âm thanh:", error);
        return false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Lấy các phần tử HTML
    const recordButton = document.getElementById('record-button');
    const fileUploadInput = document.getElementById('file-upload');
    const timerDisplay = document.getElementById('timer');
    const recordingPanel = document.getElementById('recording-panel');
    const resultsPanel = document.getElementById('results-panel');
    const resultsContent = document.getElementById('results-content');
    const resultPlayer = document.getElementById('result-player');
    const audioPlayer = resultPlayer.querySelector('audio');
    const diagnoseAgainButton = document.getElementById('diagnose-again-button');

    // Biến quản lý
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval;
    let seconds = 0; 

    if (recordButton) recordButton.addEventListener('click', toggleRecording);
    if (diagnoseAgainButton) {
        diagnoseAgainButton.addEventListener('click', () => {
            resultsPanel.style.display = 'none';
            recordingPanel.style.display = 'block';
            resetTimer();
        });
    }
    if (fileUploadInput) fileUploadInput.addEventListener('change', handleFileSelect);

    async function handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;
        event.target.value = null; 
        await handleRecordingStop(file, true);
    }

    async function toggleRecording() {
        if (isRecording) stopRecording();
        else await startRecording();
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                handleRecordingStop(audioBlob); 
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            isRecording = true;
            recordButton.classList.add('is-recording');
            startTimer();
        } catch (error) {
            console.error('Lỗi mic:', error);
            alert('Không thể truy cập micro. Kiểm tra quyền truy cập.');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            isRecording = false;
            recordButton.classList.remove('is-recording');
            stopTimer();
        }
    }
    
    function startTimer() {
        seconds = 0;
        timerDisplay.textContent = '00:00';
        timerInterval = setInterval(() => {
            seconds++;
            const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
            const secs = (seconds % 60).toString().padStart(2, '0');
            timerDisplay.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    function stopTimer() { clearInterval(timerInterval); }
    function resetTimer() { stopTimer(); seconds = 0; timerDisplay.textContent = '00:00'; }

    // --- HÀM XỬ LÝ CHÍNH ---
    async function handleRecordingStop(audioBlob, isUpload = false) {
        recordingPanel.style.display = 'none';
        resultsPanel.style.display = 'block';
        resultPlayer.style.display = 'none'; 

        if (!isUpload && seconds < 1) { 
            showError("Âm thanh quá ngắn", "Bản ghi dưới 1 giây. Vui lòng thử lại.");
            return;
        }

        resultsContent.innerHTML = '<p>Đang kiểm tra chất lượng âm thanh...</p>';
        let audioIsSilent = false; 
        if (!isUpload) audioIsSilent = await isAudioSilent(audioBlob);

        if (audioIsSilent) {
            showError("Không phát hiện âm thanh", "Không thấy tiếng ho hoặc âm thanh quá nhỏ.");
            return;
        }
        
        resultsContent.innerHTML = '<p>Đang phân tích... Vui lòng chờ trong giây lát.</p>';

        const HF_SPACE_URL = "https://nckhngt-ngt-cough-api.hf.space/";
        let predictions;
        
        try {
            const client = await Client.connect(HF_SPACE_URL);
            const result = await client.predict("/predict", { audio_file: audioBlob });
            
            // Xử lý dữ liệu trả về từ API (Hỗ trợ cả 2 định dạng phổ biến)
            const hfResultData = result.data[0];
            console.log("Dữ liệu từ API:", hfResultData);

            if (hfResultData && Array.isArray(hfResultData.confidences)) {
                predictions = hfResultData.confidences.map(item => ({
                    label: item.label,
                    confidence: item.confidence
                }));
            } else if (typeof hfResultData === 'object' && hfResultData !== null) {
                predictions = Object.entries(hfResultData)
                    .filter(([key, val]) => key !== 'label' && key !== 'confidences')
                    .map(([label, confidence]) => ({
                        label: label,
                        confidence: parseFloat(confidence)
                    }));
            } else {
                throw new Error("Không hiểu định dạng dữ liệu từ API.");
            }

            if (!predictions || predictions.length === 0) throw new Error("Không có kết quả dự đoán.");

        } catch (error) {
            console.error('Lỗi API:', error);
            resultsContent.innerHTML = `<h2>Lỗi phân tích AI</h2><p>${error.message}</p>`;
            return;
        }

        // Tìm kết quả tốt nhất
        const topPrediction = predictions.reduce((prev, current) => (prev.confidence > current.confidence) ? prev : current);
        const diagnosis_result = topPrediction.label; 
        const confidence_raw = topPrediction.confidence;

        // Gửi về Server Flask để lưu (Vẫn gửi độ tin cậy để lưu vào DB cho admin xem, nhưng không hiện lên web)
        const formData = new FormData();
        formData.append('audio_data', audioBlob);
        formData.append('diagnosis_result', diagnosis_result);
        formData.append('confidence', confidence_raw.toFixed(2)); 

        try {
            const flaskResponse = await fetch('/upload_audio', { method: 'POST', body: formData });
            const data = await flaskResponse.json();

            if (data.success) {
                // Gọi hàm hiển thị kết quả (không truyền tham số confidence nữa)
                showSuccess(data.diagnosis_result, data.filename);
            } else {
                resultsContent.innerHTML = `<h2>Lỗi máy chủ</h2><p>${data.error}</p>`;
            }
        } catch (error) {
            resultsContent.innerHTML = '<h2>Lỗi kết nối</h2><p>Không thể lưu kết quả.</p>';
        }
    }

    function showError(title, message) {
        resultsContent.innerHTML = `
            <div class="result-display warning"> 
                <div class="result-icon"><i class="fas fa-exclamation-triangle"></i></div>
                <p class="result-text-main">${title}</p>
                <p class="result-text-sub">${message}</p>
            </div>`;
        resultPlayer.style.display = 'block'; 
        audioPlayer.style.display = 'none'; 
        audioPlayer.src = '';
    }

    // --- HÀM HIỂN THỊ KẾT QUẢ ĐÃ ĐƯỢC CHỈNH SỬA ---
    function showSuccess(diagnosis, filename) {
        const classDisplayNames = {
            "healthy": "Khỏe mạnh", "asthma": "Hen suyễn", "covid": "Covid", "tuberculosis": "Bệnh lao"
        };
        const displayResult = classDisplayNames[diagnosis] || diagnosis;
        
        let iconHtml = (diagnosis === 'healthy') ? '<i class="fas fa-check-circle"></i>' : '<i class="fas fa-exclamation-triangle"></i>';
        let resultClass = (diagnosis === 'healthy') ? 'success' : 'warning';

        // Đã xóa phần hiển thị Độ tin cậy (Confidence)
        resultsContent.innerHTML = `
            <div class="result-display ${resultClass}">
                <div class="result-icon">${iconHtml}</div>
                <p class="result-text-main">${displayResult}</p>
                
                <p class="result-text-sub" style="margin-top: 15px;">Lưu ý: Kết quả chỉ mang tính chất tham khảo.</p>
            </div>`;
            
        audioPlayer.src = filename; 
        audioPlayer.style.display = 'block'; 
        resultPlayer.style.display = 'block';
    }
});