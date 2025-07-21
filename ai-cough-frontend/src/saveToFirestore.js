// saveToFirestore.js
import { db } from "./firebase";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";

export async function saveCoughRecordToFirestore(user, fileUrl, aiResult) {
  await addDoc(collection(db, "coughHistory"), {
    uid: user.uid,
    email: user.email,
    audioURL: fileUrl,
    result: aiResult,
    createdAt: serverTimestamp(),
  });
}
