document.addEventListener('DOMContentLoaded', () => {
    // --- LOGIC CHO MENU DI ĐỘNG (HAMBURGER) ---
    const hamburger = document.getElementById('hamburger');
    const navMenu = document.getElementById('nav-menu');
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });
    }

    // --- LOGIC CHO BỘ CHỌN THEME MÀU ---
    const body = document.body;
    const themePickerToggle = document.getElementById('theme-picker-toggle');
    const themePickerOptions = document.getElementById('theme-picker-options');
    const themeSwatches = document.querySelectorAll('.theme-swatch');

    // 1. Áp dụng theme đã lưu khi tải trang
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        body.dataset.theme = savedTheme;
    }

    // 2. Xử lý việc bật/tắt menu chọn màu
    if (themePickerToggle && themePickerOptions) {
        themePickerToggle.addEventListener('click', (event) => {
            event.stopPropagation();
            themePickerOptions.classList.toggle('hidden');
        });
    }

    // 3. Xử lý khi người dùng chọn một màu cụ thể
    if (themeSwatches.length > 0) {
        themeSwatches.forEach(swatch => {
            swatch.addEventListener('click', () => {
                const themeName = swatch.dataset.themeName;
                body.dataset.theme = themeName;
                localStorage.setItem('theme', themeName);
                if (themePickerOptions) {
                    themePickerOptions.classList.add('hidden');
                }
            });
        });
    }

    // 4. Ẩn menu chọn màu nếu người dùng bấm ra ngoài
    document.addEventListener('click', () => {
        if (themePickerOptions && !themePickerOptions.classList.contains('hidden')) {
            themePickerOptions.classList.add('hidden');
        }
    });

    // --- LOGIC CHO HIỆU ỨNG EQUALIZER ---
    const equalizer = document.getElementById('equalizer-container');
    if (equalizer) {
        const bars = 50; // Số lượng cột sóng
        for (let i = 0; i < bars; i++) {
            const bar = document.createElement('div');
            bar.classList.add('bar');
            equalizer.appendChild(bar);
        }

        setInterval(() => {
            document.querySelectorAll('.bar').forEach(bar => {
                bar.style.height = `${Math.random() * 80 + 5}%`;
            });
        }, 150);
    }

    // --- LOGIC MỚI CHO HIỆN/ẨN MẬT KHẨU ---
    const togglePassword = document.getElementById('togglePassword');
    const passwordInput = document.getElementById('password');

    if (togglePassword && passwordInput) {
        togglePassword.addEventListener('click', function () {
            // Chuyển đổi thuộc tính 'type' của ô input
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            
            // Chuyển đổi icon con mắt
            this.classList.toggle('fa-eye-slash');
        });
    }
});