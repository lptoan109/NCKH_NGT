document.addEventListener('DOMContentLoaded', () => {
    // Lấy các phần tử DOM
    const recordButton = document.getElementById('record-button');
    const timerElement = document.getElementById('timer');
    const recordingPanel = document.getElementById('recording-panel');
    const resultsPanel = document.getElementById('results-panel');
    const resultMessage = document.getElementById('result-message');
    const resultPlayerContainer = document.getElementById('result-player');
    const audioPlayer = resultPlayerContainer ? resultPlayerContainer.querySelector('audio') : null;

    // Thoát nếu không tìm thấy các phần tử cần thiết
    if (!recordButton || !timerElement || !recordingPanel || !resultsPanel || !audioPlayer) {
        console.error("Một hoặc nhiều phần tử không được tìm thấy trên trang.");
        return; 
    }

    let mediaRecorder;
    let audioChunks = [];
    let timerInterval;
    let seconds = 0;
    let isRecording = false;

    // Gán sự kiện click cho nút ghi âm
    recordButton.onclick = async () => {
        if (!isRecording) {
            // --- BẮT ĐẦU GHI ÂM ---
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                isRecording = true;
                recordButton.classList.add('is-recording');
                startTimer();

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    uploadAudio(audioBlob);
                    audioChunks = [];
                };

            } catch (error) {
                console.error("Lỗi khi truy cập micro:", error);
                alert("Không thể truy cập micro. Vui lòng cấp quyền và thử lại.");
            }
        } else {
            // --- DỪNG GHI ÂM ---
            mediaRecorder.stop();
            isRecording = false;
            recordButton.classList.remove('is-recording');
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
        // Chuyển đổi giao diện
        recordingPanel.style.display = 'none';
        resultsPanel.style.display = 'block';
        resultMessage.textContent = 'Đang phân tích... Vui lòng chờ trong giây lát.';
        resultPlayerContainer.style.display = 'none';

        const formData = new FormData();
        formData.append('audio_data', audioBlob, 'recording.wav');

        try {
            const response = await fetch('/upload_audio', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            if (result.success) {
                resultMessage.textContent = 'Phân tích thành công!';
                const audioUrl = `/static/uploads/${result.filename}`;
                audioPlayer.src = audioUrl;
                resultPlayerContainer.style.display = 'block';
            } else {
                resultMessage.textContent = 'Đã có lỗi xảy ra khi tải file lên.';
            }
        } catch (error) {
            console.error('Lỗi khi tải file lên:', error);
            resultMessage.textContent = 'Lỗi mạng. Không thể kết nối đến server.';
        }
    }
});