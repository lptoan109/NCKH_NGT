// recorder.js

// Import thư viện Gradio Client trực tiếp ở đầu file
import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

// --- HÀM TRỢ GIÚP: Chuyển Blob sang Base64 ---
function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => {
            const base64data = reader.result;
            resolve(base64data);
        };
        reader.onerror = error => reject(error);
    });
}

// --- HÀM TRỢ GIÚP MỚI: Kiểm tra file âm thanh có bị im lặng không ---
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
        console.error("Không thể phân tích âm thanh:", error);
        return false;
    }
}
// -------------------------------------------


document.addEventListener('DOMContentLoaded', () => {
    // Lấy các phần tử HTML cần thiết
    const recordButton = document.getElementById('record-button');
    const fileUploadInput = document.getElementById('file-upload'); // <-- THÊM DÒNG NÀY
    const timerDisplay = document.getElementById('timer');
    const recordingPanel = document.getElementById('recording-panel');
    const resultsPanel = document.getElementById('results-panel');
    const resultsContent = document.getElementById('results-content');
    const resultPlayer = document.getElementById('result-player');
    const audioPlayer = resultPlayer.querySelector('audio');
    const diagnoseAgainButton = document.getElementById('diagnose-again-button');

    // Các biến để quản lý trạng thái ghi âm
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval;
    let seconds = 0; // Biến này quan trọng để kiểm tra thời lượng

    if (recordButton) {
        recordButton.addEventListener('click', toggleRecording);
    }

    if (diagnoseAgainButton) {
        diagnoseAgainButton.addEventListener('click', () => {
            resultsPanel.style.display = 'none';
            recordingPanel.style.display = 'block';
            resetTimer();
        });
    }

    // --- BỘ XỬ LÝ SỰ KIỆN MỚI CHO TẢI FILE ---
    if (fileUploadInput) {
        fileUploadInput.addEventListener('change', handleFileSelect);
    }

    async function handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) {
            return; // Người dùng hủy
        }

        // Đặt lại giá trị input để họ có thể upload file y hệt lần nữa
        event.target.value = null; 

        // File chính là một Blob, gọi hàm xử lý và báo đây là file upload
        await handleRecordingStop(file, true);
    }
    // --- HẾT BỘ XỬ LÝ MỚI ---


    async function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                
                // Gọi hàm xử lý, isUpload sẽ mặc định là false
                handleRecordingStop(audioBlob); 

                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            isRecording = true;
            recordButton.classList.add('is-recording');
            startTimer();

        } catch (error) {
            console.error('Lỗi khi truy cập micro:', error);
            alert('Không thể truy cập micro. Vui lòng kiểm tra lại quyền truy cập trong cài đặt của trình duyệt.');
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
    
    // --- Các hàm quản lý đồng hồ (giữ nguyên) ---
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

    function stopTimer() {
        clearInterval(timerInterval);
    }

    function resetTimer() {
        stopTimer();
        seconds = 0;
        timerDisplay.textContent = '00:00';
    }


    // --- HÀM XỬ LÝ ĐÃ SỬA: Thêm tham số "isUpload" ---
    async function handleRecordingStop(audioBlob, isUpload = false) {
        
        // --- BẮT ĐẦU PHẦN KIỂM TRA ---

        // 1. Hiển thị bảng kết quả và ẩn bảng ghi âm
        recordingPanel.style.display = 'none';
        resultsPanel.style.display = 'block';
        resultPlayer.style.display = 'none'; 

        // 2. Kiểm tra thời lượng (CHỈ ÁP DỤNG CHO GHI ÂM)
        // 'seconds' là biến toàn cục của đồng hồ
        if (!isUpload && seconds < 1) { 
            console.log("Validation Lỗi: Âm thanh quá ngắn");
            resultsContent.innerHTML = `
                <div class="result-display warning"> 
                    <div class="result-icon"><i class="fas fa-exclamation-triangle"></i></div>
                    <p class="result-text-main">Âm thanh quá ngắn</p>
                    <p class="result-text-sub">Bản ghi âm của bạn dưới 1 giây. Vui lòng ghi âm lại và ho rõ ràng hơn.</p>
                </div>`;
            
            resultPlayer.style.display = 'block'; 
            audioPlayer.style.display = 'none'; 
            audioPlayer.src = '';
            return; // Dừng hàm, không gọi API
        }

        // 3. Kiểm tra file có bị im lặng không (Áp dụng cho cả 2)
        resultsContent.innerHTML = '<p>Đang kiểm tra chất lượng âm thanh...</p>';
        
        const audioIsSilent = await isAudioSilent(audioBlob);
        if (audioIsSilent) {
            console.log("Validation Lỗi: Âm thanh quá im lặng");
            resultsContent.innerHTML = `
                <div class="result-display warning">
                    <div class="result-icon"><i class="fas fa-exclamation-triangle"></i></div>
                    <p class="result-text-main">Không phát hiện âm thanh</p>
                    <p class="result-text-sub">Không phát hiện thấy tiếng ho hoặc âm thanh quá nhỏ. Vui lòng ghi âm lại hoặc tải lên file khác rõ ràng hơn.</p>
                </div>`;
            
            resultPlayer.style.display = 'block';
            audioPlayer.style.display = 'none';
            audioPlayer.src = '';
            return; // Dừng hàm, không gọi API
        }
        
        // --- KẾT THÚC PHẦN KIỂM TRA ---

        // Nếu vượt qua, tiếp tục gọi API
        resultsContent.innerHTML = '<p>Đang phân tích... Vui lòng chờ trong giây lát.</p>';

        // 3. Gọi API Hugging Face bằng @gradio/client
        const HF_SPACE_URL = "https://nckhngt-ngt-cough-api.hf.space/";

        let hfResultData;
        let resultObject; 
        try {
            const client = await Client.connect(HF_SPACE_URL);

            const result = await client.predict("/predict", {
                audio_file: audioBlob 
            });

            hfResultData = result.data;
            
            if (!Array.isArray(hfResultData) || hfResultData.length === 0) {
                console.error("Dữ liệu trả về không phải mảng hoặc rỗng:", hfResultData);
                throw new Error("Kết quả API trả về có cấu trúc không mong đợi (không phải mảng).");
            }

            resultObject = hfResultData[0]; 

            if (!resultObject || !resultObject.confidences) {
                console.error("Đối tượng kết quả không có 'confidences':", resultObject);
                throw new Error("Kết quả API không hợp lệ hoặc thiếu 'confidences'.");
            }

        } catch (error) {
            console.error('Lỗi khi gọi API Hugging Face:', error);
            resultsContent.innerHTML = `<h2>Đã có lỗi xảy ra</h2><p>Không thể kết nối tới máy chủ AI. Vui lòng thử lại sau.</p><p style="font-size: 0.8em; color: var(--text-light);">${error.message}</p>`;
            return;
        }

        // 4. Lấy kết quả chẩn đoán và độ tin cậy
        const predictions = resultObject.confidences;
        
        const topPrediction = predictions.reduce((prev, current) => (prev.confidence > current.confidence) ? prev : current);
        const diagnosis_result = topPrediction.label; 
        
        const confidence_raw = topPrediction.confidence;
        const confidence_display = (confidence_raw * 100).toFixed(0); 


        // 5. Gửi file âm thanh VÀ kết quả về server Flask để lưu
        const formData = new FormData();
        formData.append('audio_data', audioBlob);
        formData.append('diagnosis_result', diagnosis_result);
        formData.append('confidence', confidence_raw.toFixed(2)); 

        try {
            // Gọi về server Flask (trên PythonAnywhere)
            const flaskResponse = await fetch('/upload_audio', { method: 'POST', body: formData });
            const data = await flaskResponse.json();

            if (data.success) {
                let iconHtml = '';
                let resultClass = '';
                
                const classDisplayNames = {
                    "healthy": "Khỏe mạnh",
                    "asthma": "Hen suyễn",
                    "covid": "Covid",
                    "tuberculosis": "Bệnh lao"
                };
                
                const displayResult = classDisplayNames[data.diagnosis_result] || data.diagnosis_result;

                if (data.diagnosis_result === 'healthy') {
                    iconHtml = '<i class="fas fa-check-circle"></i>';
                    resultClass = 'success';
                } else {
                    iconHtml = '<i class="fas fa-exclamation-triangle"></i>';
                    resultClass = 'warning';
                }
                
                const resultHtml = `
                    <div class="result-display ${resultClass}">
                        <div class="result-icon">${iconHtml}</div>
                        <p class="result-text-main">${displayResult}</p>
                        
                        <p style="color: var(--text-light); margin-top: 5px; font-size: 1em;">
                            Độ tin cậy: <strong>${confidence_display}%</strong>
                        </p>
                        
                        <p class="result-text-sub" style="margin-top: 15px;">Lưu ý: Kết quả chỉ mang tính chất tham khảo.</p>
                    </div>
                `;
                resultsContent.innerHTML = resultHtml;
                audioPlayer.src = data.filename; 
                audioPlayer.style.display = 'block'; 
                resultPlayer.style.display = 'block';

            } else {
                resultsContent.innerHTML = `<h2>Đã có lỗi xảy ra</h2><p>${data.error || 'Không thể lưu kết quả.'}</p>`;
            }
        } catch (error) {
            console.error('Lỗi khi gửi kết quả về server Flask:', error);
            resultsContent.innerHTML = '<h2>Đã có lỗi xảy ra</h2><p>Lỗi kết nối tới server (để lưu file).</p>';
        }
    }
});