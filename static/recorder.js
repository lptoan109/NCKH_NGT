document.addEventListener('DOMContentLoaded', () => {
    const recordButton = document.getElementById('record-button');
    const timerElement = document.getElementById('timer');
    const recordingPanel = document.getElementById('recording-panel');
    const resultsPanel = document.getElementById('results-panel');
    const resultMessage = document.getElementById('result-message');
    const resultPlayer = document.getElementById('result-player');
    const audioPlayer = resultPlayer.querySelector('audio');
    const diagnoseAgainButton = document.getElementById('diagnose-again-button');

    if (!recordButton) return;

    let mediaRecorder;
    let audioChunks = [];
    let timerInterval;
    let seconds = 0;

    // Hàm cập nhật đồng hồ
    function updateTimer() {
        seconds++;
        const mins = String(Math.floor(seconds / 60)).padStart(2, '0');
        const secs = String(seconds % 60).padStart(2, '0');
        timerElement.textContent = `${mins}:${secs}`;
    }

    // Hàm bắt đầu ghi âm
    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                sendAudioToServer(audioBlob);
                audioChunks = []; // Reset lại
            };

            mediaRecorder.start();
            recordButton.classList.add('is-recording');
            seconds = 0;
            timerElement.textContent = '00:00';
            timerInterval = setInterval(updateTimer, 1000);
        } catch (error) {
            console.error("Error accessing microphone:", error);
            alert("Không thể truy cập micro. Vui lòng cấp quyền và thử lại.");
        }
    }

    // Hàm dừng ghi âm
    function stopRecording() {
        mediaRecorder.stop();
        recordButton.classList.remove('is-recording');
        clearInterval(timerInterval);
    }

    // Hàm gửi audio lên server và hiển thị kết quả
    async function sendAudioToServer(audioBlob) {
        // Hiển thị panel kết quả với thông báo đang chờ
        recordingPanel.style.display = 'none';
        resultsPanel.style.display = 'block';
        resultMessage.innerHTML = '<h2>Đang phân tích...</h2><p>Vui lòng chờ trong giây lát.</p>';
        resultPlayer.style.display = 'none';

        const formData = new FormData();
        formData.append('audio_data', audioBlob);

        try {
            const response = await fetch('/upload_audio', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                // Cập nhật giao diện với kết quả nhận được
                const resultHtml = `
                    <h2>Kết quả: <span style="color: var(--primary-color);">${data.diagnosis_result}</span></h2>
                    <p>Đây là kết quả phân tích dựa trên tiếng ho của bạn. Vui lòng lưu ý kết quả chỉ mang tính tham khảo.</p>
                `;
                resultMessage.innerHTML = resultHtml;
                audioPlayer.src = data.filename;
                resultPlayer.style.display = 'block';
            } else {
                resultMessage.innerHTML = '<h2>Đã có lỗi xảy ra</h2><p>Không thể phân tích file âm thanh. Vui lòng thử lại.</p>';
            }
        } catch (error) {
            console.error('Error uploading audio:', error);
            resultMessage.innerHTML = '<h2>Đã có lỗi xảy ra</h2><p>Lỗi kết nối tới server. Vui lòng thử lại.</p>';
        }
    }

    // Gán sự kiện cho các nút
    recordButton.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            startRecording();
        }
    });

    diagnoseAgainButton.addEventListener('click', () => {
        resultsPanel.style.display = 'none';
        recordingPanel.style.display = 'block';
        seconds = 0;
        timerElement.textContent = '00:00';
    });
});