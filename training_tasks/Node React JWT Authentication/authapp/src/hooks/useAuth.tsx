// src/hooks/useAuth.tsx
import { useContext } from "react";
import AuthContext from "../contexts/AuthContext";

const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context; // Ensure we return the full context
};

export default useAuth;
