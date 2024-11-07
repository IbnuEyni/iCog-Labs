// routes/auth.js
const express = require("express");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const User = require("../models/user");
const authenticateJWT = require("../middleware/authenticate");
const router = express.Router();

// Register Route
router.post("/register", async (req, res) => {
  try {
    const hashedPassword = await bcrypt.hash(req.body.password, 10);
    const user = new User({
      name: req.body.name,
      email: req.body.email,
      password: hashedPassword,
    });
    await user.save();
    res.status(201).json({ message: "User registered successfully" });
  } catch (error) {
    console.log(error);
    res.status(400).json({ error: "Registration failed" });
  }
});

// Login Route
// routes/auth.js
router.post("/login", async (req, res) => {
    try {
      const user = await User.findOne({ email: req.body.email });
      
      if (user && await user.comparePassword(req.body.password)) {
        // Generate access token (short expiration time)
        const accessToken = jwt.sign(
          { id: user._id },
          process.env.JWT_SECRET,
          { expiresIn: "15m" }  // Set to 15 minutes or any short duration you prefer
        );
  
        // Generate refresh token (long expiration time)
        const refreshToken = jwt.sign(
          { id: user._id },
          process.env.JWT_REFRESH_SECRET,
          { expiresIn: "7d" }  // Set to 7 days or any long duration you prefer
        );
  
        res.json({ accessToken, refreshToken });
      } else {
        res.status(401).json({ error: "Invalid credentials" });
      }
    } catch (error) {
        console.log(error)
      res.status(500).json({ error: "Server error" });
    }
  });
  

// Protected Route
router.get("/profile", authenticateJWT, async (req, res) => {
    try {
      const user = await User.findById(req.user.id).select("-password");
      if (!user) return res.sendStatus(404);
      res.json(user);
    } catch (error) {
      res.status(500).json({ error: "Server error" });
    }
  });

module.exports = router;
