# Full-Stack JWT Authentication with Express.js and React (Vite)

This project is a full-stack web application that implements a secure authentication system using JSON Web Tokens (JWT). The backend is powered by **Express.js** and the frontend by **React** with **Vite**, providing a fast and modern development experience. This setup allows users to register, log in, and access protected routes, offering a solid foundation for projects requiring user authentication.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Contributing](#contibuting)

## Features

- **User Registration**: Sign up new users with validation
- **User Login**: Authenticate registered users and issue JWT
- **Protected Routes**: Restrict access to routes using JWT
- **JWT-based Authentication**: Store JWT securely and use it to access restricted resources
- **Logout Functionality**: Clear user session and revoke access

## Technologies Used

- **Backend**: Node.js, Express.js, JSON Web Tokens (JWT), bcrypt.js (for password hashing)
- **Frontend**: React, Vite, Axios (for API requests)
- **Database**: MongoDB (optional, for storing user credentials)

## Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

- **Node.js** (v14 or higher)
- **MongoDB** (optional, for local development)
- **pnpm** (install via npm if you don't have it)
  ```bash
  npm install -g pnpm
  ```

### Installation
1. clone the repository
    ```bash
    git clone https://github.com/IbnuEyni/node-react-JWT-authentication.git
    cd node-react-JWT-authentication
    ```
2. Install backend dependencies:
    ```bash
    cd backend
    npm install
    ```
3. Install frontend dependencies:
    ```bash
    cd authapp
    pnpm install
    ```
### Running the Application

1. Start the backend server:
    ```bash
    cd backend
    npm start
    ```
2. Start the frontend application:
    ```bash
    cd authapp
    pnpm dev
    ```
3. Open your browser and navigate to the specified port.

### Project Structure

    project-root
    ├── authapp         # React frontend
    │   ├── src         # Source files
    │   └── ...
    ├── backend         # Express backend
    │   ├── config      # Configuration files
    │   ├── controllers # API controllers
    │   ├── models      # Database models
    │   ├── routes      # API routes
    │   └── ...
    └── ...
    
### Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:

    ```bash
    git checkout -b feature/my-feature
    ```

3. Make your changes and commit them:

   ```bash
    git commit -m "Add some feature"
    ```

4. Push to the branch:

    ```bash
    git push origin feature/my-feature
    ```

5. Open a Pull Request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.