# Neo4j Projects: Social Network App

This repository includes two distinct projects utilizing Neo4j and Python: a Neo4j Resume Graph and a Neo4j Social Network Application. The first project transforms a resume into a Neo4j knowledge graph, while the second develops a social network application with CLI interactions.

## Project Overview

#### Neo4j Social Network App: 
Implements a CLI-based social network application with user management, friend interactions, and more.


## Neo4j Social Network App

### Description: 
Developed a social network application using Neo4j to handle various user interactions and data management tasks. The application is designed to facilitate user registration, profile management, and social interactions such as friendships, posts, comments, and group activities.

## Key Features:

#### User Registration:
Added users to the network with attributes like name, age, location, and interests.

#### Profile Management:
Updated user information, including age, location, and interests.

#### Friendship Management:
Enabled users to add and remove friends.
Sent, accepted, or rejected friend requests.

#### Post Management:
Created and managed user posts.
Allowed users to like and comment on posts.

#### Group Management:
Created and managed user groups.
Enabled users to join groups and view group members.

#### Friend Recommendations:
Provided recommendations for new friends based on mutual connections.

#### User Search:
Implemented search functionality to find users based on name, location, and interests.

# Development  

#### Connect to Neo4j:
Use the neo4j Python package to interact with the database.
Implement functions for managing users, friendships, and interactions.

#### CLI Interface:
Implement a command-line interface to interact with the application.
Provide commands for user management, friend requests, and more.

## Python Implementation

Neo4j Package: Use the neo4j package for database interactions.
Function Implementation: Implement functions for each feature (e.g., create_user(), send_friend_request()).
Error Handling: Implement robust error handling to manage database interactions and user inputs.

## Getting Started

Clone the Repository:

    git clone https://github.com/IbnuEyni/iCog-Labs.git

### Prerequisites

Install Neo4j and set it up on your local machine.
Install the required Python packages using:

    pip install -r requirements.txt

Neo4j Configuration

Create a .env file in the root directory of your project.

Add your Neo4j credentials to the .env file using the following format:

    NEO4J_USERNAME=your-username
    NEO4J_PASSWORD=your-password

Ensure that your .env file is included in your .gitignore to avoid exposing sensitive credentials.

Running the Application

After configuring your environment, you can run the application with the CLI interface for only for Social Network App.

## Contributing

We welcome contributions to this project! To contribute, please follow these steps:

Fork the Repository:
Click the "Fork" button at the top right of this page to create your own copy of the repository.

Create a New Branch:

    git checkout -b new_feature

Make Your Changes:

Implement your feature or fix in the new branch.
Commit Your Changes:

    git add .
    git commit -m "Add new feature"
Push to Your Forked Repository:

    git push origin new_feature
Create a Pull Request:
Go to the "Pull Requests" tab of the original repository and click "New Pull Request."
Select your new_feature branch and submit the pull request.
