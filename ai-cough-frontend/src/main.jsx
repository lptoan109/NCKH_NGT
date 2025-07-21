
// ReactDOM.createRoot(document.getElementById("root")).render(<RouterApp />);
import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Landing from "./Landing.jsx";
import Home from "./Home.jsx";
import App from "./App.jsx"; // Nếu bạn đổi tên thành Diagnose.jsx thì sửa tại đây
import History from "./History.jsx";
import { getAuth, onAuthStateChanged } from "firebase/auth";

function RouterApp() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser);
    });
    return () => unsubscribe();
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/home" element={<Home />} />
        <Route path="/diagnose" element={<App />} />
        <Route path="/history" element={<History user={user} />} />
      </Routes>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<RouterApp />);
