import axios from "axios";

const API = axios.create({
    baseURL: "https://smartdefectai.onrender.com"
});

export default API;