import { useState } from "react";
import API from "../services/api";

import { useDropzone } from "react-dropzone";

import { motion } from "framer-motion";

import "../styles/home.css";


export default function Home() {

    const [file, setFile] = useState(null);

    const [prediction, setPrediction] = useState("");

    const [confidence, setConfidence] = useState("");

    const [loading, setLoading] = useState(false);


    const onDrop = acceptedFiles => {

        setFile(acceptedFiles[0]);
    };


    const { getRootProps, getInputProps } = useDropzone({
        onDrop
    });


    const handleUpload = async () => {

        if (!file) return;

        const formData = new FormData();

        formData.append("file", file);

        try {

            setLoading(true);

            const response = await API.post(
    "/predict",
    formData,
    {
        headers: {
            "Content-Type": "multipart/form-data"
        }
    }
);

console.log(response.data);


            setPrediction(response.data.prediction);

            setConfidence(response.data.confidence);

        } catch (error) {

            console.error(error.response?.data || error.message);

        } finally {

            setLoading(false);
        }
    };


    return (

        <div className="container">

            <motion.h1
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                SmartDefectAI
            </motion.h1>

            <div {...getRootProps()} className="dropzone">

                <input {...getInputProps()} />

                <p>
                    Drag & Drop Defect Image Here
                </p>

            </div>

            {file && (
                <p className="filename">
                    {file.name}
                </p>
            )}

            <button onClick={handleUpload}>

                Analyze Defect

            </button>

            {loading && <p>Analyzing...</p>}

            {prediction && (

                <motion.div
                    className="result"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                >

                    <h2>Prediction</h2>

                    <p>{prediction}</p>

                    <h3>
                        Confidence: {confidence}%
                    </h3>

                </motion.div>
            )}
        </div>
    );
}