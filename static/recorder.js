document.addEventListener('DOMContentLoaded', () => {
    const recordButton = document.getElementById('record-button');
    const timerElement = document.getElementById('timer');
    
    // Thêm các panel mới
    const recordingPanel = document.getElementById('recording-panel');
    const resultsPanel = document.getElementById('results-panel');

    // Kiểm tra xem các phần tử có tồn tại trên trang không
    if (!recordButton || !timerElement || !recordingPanel || !resultsPanel) {
        return; // Thoát nếu không phải trang chẩn đoán
    }

    let mediaRecorder;
    let audioChunks = [];
    let timerInterval;
    let seconds = 0;
    let isRecording = false;

    recordButton.onclick = async () => {
        if (!isRecording) {
            // --- BẮT ĐẦU GHI ÂM ---
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    // Gửi file đến server
                    uploadAudio(audioBlob);

                    // Reset
                    audioChunks = [];
                };

                mediaRecorder.start();
                isRecording = true;
                recordButton.classList.add('is-recording'); // Thêm class để có hiệu ứng
                startTimer();

            } catch (error) {
                console.error("Lỗi khi truy cập micro:", error);
                alert("Không thể truy cập micro. Vui lòng cấp quyền cho trang web.");
            }
        } else {
            // --- DỪNG GHI ÂM ---
            mediaRecorder.stop();
            isRecording = false;
            recordButton.classList.remove('is-recording'); // Bỏ class hiệu ứng
            stopTimer();
        }
    };

    function startTimer() {
        seconds = 0;
        timerElement.textContent = '00:00';
        timerInterval = setInterval(() => {
            seconds++;
            const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
            const secs = (seconds % 60).toString().padStart(2, '0');
            timerElement.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    function stopTimer() {
        clearInterval(timerInterval);
    }

    async function uploadAudio(audioBlob) {
        // Hiển thị panel kết quả với thông báo "Đang phân tích..."
        recordingPanel.style.display = 'none';
        resultsPanel.style.display = 'block';

        const formData = new FormData();
        formData.append('audio_data', audioBlob, 'recording.wav');

        try {
            const response = await fetch('/upload_audio', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            if (result.success) {
                console.log('Tải lên thành công:', result.filename);
                // Sau này, bạn sẽ hiển thị kết quả chẩn đoán thật ở đây
                resultsPanel.querySelector('.results-content p').textContent = 'Phân tích thành công! Kết quả sẽ sớm được hiển thị.';
            } else {
                resultsPanel.querySelector('.results-content p').textContent = 'Đã có lỗi xảy ra khi tải file lên.';
            }
        } catch (error) {
            console.error('Lỗi khi tải file lên:', error);
            resultsPanel.querySelector('.results-content p').textContent = 'Lỗi mạng. Không thể kết nối đến server.';
        }
    }
});