// // src/uploadCloudinary.js
// export async function uploadToCloudinary(file) {
//   const formData = new FormData();
//   formData.append("file", file);
//   formData.append("upload_preset", "your_upload_preset"); // Thay bằng upload preset của bạn
//   formData.append("cloud_name", "your_cloud_name"); // Thay bằng tên cloud

//   const res = await fetch("https://api.cloudinary.com/v1_1/your_cloud_name/auto/upload", {
//     method: "POST",
//     body: formData,
//   });

//   const data = await res.json();
//   return data.secure_url;
// }
// uploadCloudinary.js
import axios from 'axios';

export async function uploadToCloudinary(file) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("upload_preset", "NGTcoughspectron");

  const response = await axios.post("https://api.cloudinary.com/v1_1/dbkmss0ko/audio/upload", formData);
  return response.data.secure_url;
}
