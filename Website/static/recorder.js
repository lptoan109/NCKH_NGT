// recorder.js

// --- HÀM TRỢ GIÚP: Chuyển Blob sang Base64 ---
// Cần thiết để gửi file âm thanh tới API Gradio
function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => {
            // Loại bỏ phần đầu "data:audio/wav;base64,"
            // const base64data = reader.result.split(',')[1];
            // Sửa lỗi: Giữ nguyên phần đầu "data:..." vì API Gradio cần nó
            const base64data = reader.result;
            resolve(base64data);
        };
        reader.onerror = error => reject(error);
    });
}
// -------------------------------------------


document.addEventListener('DOMContentLoaded', () => {
    // Lấy các phần tử HTML cần thiết từ trang diagnose.html
    const recordButton = document.getElementById('record-button');
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
    let seconds = 0;

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
                
                // --- THAY ĐỔI LỚN BẮT ĐẦU TỪ ĐÂY ---
                // Thay vì gọi sendAudioToServer, chúng ta gọi hàm mới
                // để xử lý logic API Hugging Face
                handleRecordingStop(audioBlob); 
                // -----------------------------------

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

    // --- HÀM XỬ LÝ MỚI KHI GHI ÂM DỪNG ---
    async function handleRecordingStop(audioBlob) {
        // 1. Hiển thị bảng kết quả và thông báo chờ
        recordingPanel.style.display = 'none';
        resultsPanel.style.display = 'block';
        resultsContent.innerHTML = '<p>Đang phân tích... Vui lòng chờ trong giây lát.</p>'; 
        resultPlayer.style.display = 'none';

        // 2. Chuyển Blob sang Base64
        let base64Audio;
        try {
            base64Audio = await blobToBase64(audioBlob);
        } catch (error) {
            console.error('Lỗi chuyển đổi Base64:', error);
            resultsContent.innerHTML = '<h2>Đã có lỗi xảy ra</h2><p>Lỗi xử lý file âm thanh.</p>';
            return;
        }

        // 3. Gọi API Hugging Face
        // *** ĐÂY LÀ DÒNG ĐÃ SỬA ***
        const HF_API_URL = "https://nckhNGT-NGT-cough-api.hf.space/run/predict";

        let hfResultData;
        try {
            const hfResponse = await fetch(HF_API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    "data": [
                        // Gradio API cần toàn bộ chuỗi base64, bao gồm cả "data:audio/wav;base64,"
_                        // (Đã sửa lại hàm blobToBase64 để trả về đầy đủ)
_                        base64Audio
                    ]
                })
            });

            hfResultData = await hfResponse.json();

            if (!hfResultData.data || !hfResultData.data[0]) {
                throw new Error("Kết quả trả về từ API không hợp lệ.");
            }

        } catch (error) {
            console.error('Lỗi khi gọi API Hugging Face:', error);
            resultsContent.innerHTML = `<h2>Đã có lỗi xảy ra</h2><p>Không thể kết nối tới máy chủ AI. Vui lòng thử lại sau.</p><p style="font-size: 0.8em; color: var(--text-light);">${error.message}</p>`;
            return;
        }

        // 4. Lấy kết quả chẩn đoán và độ tin cậy
        const predictions = hfResultData.data[0].confidences;
        const topPrediction = predictions.reduce((prev, current) => (prev.confidence > current.confidence) ? prev : current);
        
        const diagnosis_result = topPrediction.label; // vd: "healthy"
        const confidence = topPrediction.confidence.toFixed(2); // vd: "0.85"

        // 5. Gửi file âm thanh VÀ kết quả về server Flask để lưu
        const formData = new FormData();
        formData.append('audio_data', audioBlob);
        formData.append('diagnosis_result', diagnosis_result);
        formData.append('confidence', confidence);

        try {
            // Gọi về server Flask (đã được sửa)
            const flaskResponse = await fetch('/upload_audio', { method: 'POST', body: formData });
            const data = await flaskResponse.json();

            if (data.success) {
                let iconHtml = '';
                let resultClass = '';
                
                // Cập nhật tên lớp (phải khớp với tên lớp trên Hugging Face)
                const classDisplayNames = {
                    "healthy": "Khỏe mạnh",
                    "asthma": "Hen suyễn",
                    "covid": "Covid",
                    "tuberculosis": "Bệnh lao"
                };
                
      _         const displayResult = classDisplayNames[data.diagnosis_result] || data.diagnosis_result;

                if (data.diagnosis_result === 'healthy') {
                    iconHtml = '<i class="fas fa-check-circle"></i>';
                    resultClass = 'success';
                } else {
                    iconHtml = '<i class="fas fa-exclamation-triangle"></i>';
                    resultClass = 'warning';
          _       }
                
                const resultHtml = `
                    <div class="result-display ${resultClass}">
                        <div class="result-icon">${iconHtml}</div>
                        <p class="result-text-main">${displayResult}</p>
                        <p class="result-text-sub">Lưu ý: Kết quả chỉ mang tính chất tham khảo.</p>
                    </div>
                `;
                resultsContent.innerHTML = resultHtml;
                audioPlayer.src = data.filename; // Flask trả về đường dẫn file đã lưu
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