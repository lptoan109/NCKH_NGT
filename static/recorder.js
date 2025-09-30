

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

    // Kiểm tra xem các phần tử có tồn tại không trước khi gán sự kiện
    if (recordButton) {
        recordButton.addEventListener('click', toggleRecording);
    }

    if (diagnoseAgainButton) {
        diagnoseAgainButton.addEventListener('click', () => {
            // Reset giao diện về trạng thái ban đầu
            resultsPanel.style.display = 'none';
            recordingPanel.style.display = 'block';
            resetTimer();
        });
    }

    // Hàm chính để Bắt đầu/Dừng ghi âm
    async function toggleRecording() {
        if (isRecording) {
            // Nếu đang ghi âm -> Dừng lại
            stopRecording();
        } else {
            // Nếu chưa ghi âm -> Bắt đầu
            await startRecording();
        }
    }

    // Hàm bắt đầu quá trình ghi âm
    async function startRecording() {
        try {
            // Yêu cầu quyền truy cập micro
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Khởi tạo MediaRecorder
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = []; // Xóa các đoạn âm thanh cũ

            // Sự kiện này được gọi khi có một đoạn âm thanh mới
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            // Sự kiện này được gọi khi việc ghi âm kết thúc
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                // Gọi hàm để gửi file âm thanh lên server
                sendAudioToServer(audioBlob);
                // Dừng track micro để tắt icon trên trình duyệt
                stream.getTracks().forEach(track => track.stop());
            };

            // Bắt đầu ghi âm
            mediaRecorder.start();
            isRecording = true;
            recordButton.classList.add('is-recording'); // Thêm class để có hiệu ứng animation
            startTimer();

        } catch (error) {
            console.error('Lỗi khi truy cập micro:', error);
            alert('Không thể truy cập micro. Vui lòng kiểm tra lại quyền truy cập trong cài đặt của trình duyệt.');
        }
    }

    // Hàm dừng quá trình ghi âm
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            isRecording = false;
            recordButton.classList.remove('is-recording');
            stopTimer();
        }
    }
    
    // --- Các hàm quản lý đồng hồ đếm giờ ---
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
});



async function sendAudioToServer(audioBlob) {
    // Lấy các phần tử HTML một lần nữa vì chúng nằm ngoài phạm vi của DOMContentLoaded
    const recordingPanel = document.getElementById('recording-panel');
    const resultsPanel = document.getElementById('results-panel');
    const resultsContent = document.getElementById('results-content');
    const resultPlayer = document.getElementById('result-player');
    const audioPlayer = resultPlayer.querySelector('audio');

    recordingPanel.style.display = 'none';
    resultsPanel.style.display = 'block';
    resultsContent.innerHTML = '<p>Đang phân tích... Vui lòng chờ trong giây lát.</p>'; // Thông báo chờ
    resultPlayer.style.display = 'none';

    const formData = new FormData();
    formData.append('audio_data', audioBlob);

    // Lấy theme hiện tại từ localStorage
    const currentTheme = localStorage.getItem('theme') || 'default';
    formData.append('theme', currentTheme);

    try {
        const response = await fetch('/upload_audio', { method: 'POST', body: formData });
        const data = await response.json();

        if (data.success) {
            let iconHtml = '';
            let resultClass = '';

            // Chọn icon và màu sắc dựa trên kết quả
            if (data.diagnosis_result === 'Khỏe mạnh') {
                iconHtml = '<i class="fas fa-check-circle"></i>';
                resultClass = 'success';
            } else {
                iconHtml = '<i class="fas fa-exclamation-triangle"></i>';
                resultClass = 'warning';
            }
            
            // Tạo giao diện kết quả mới
            const resultHtml = `
                <div class="result-display ${resultClass}">
                    <div class="result-icon">${iconHtml}</div>
                    <p class="result-text-main">${data.diagnosis_result}</p>
                    <p class="result-text-sub">Lưu ý: Kết quả chỉ mang tính chất tham khảo.</p>
                </div>
            `;
            resultsContent.innerHTML = resultHtml;
            audioPlayer.src = data.filename;
            resultPlayer.style.display = 'block';

        } else {
            resultsContent.innerHTML = '<h2>Đã có lỗi xảy ra</h2><p>Không thể phân tích file âm thanh.</p>';
        }
    } catch (error) {
        console.error('Error uploading audio:', error);
        resultsContent.innerHTML = '<h2>Đã có lỗi xảy ra</h2><p>Lỗi kết nối tới server.</p>';
    }
}