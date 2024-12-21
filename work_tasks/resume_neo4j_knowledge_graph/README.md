# Neo4j Projects: Resume Graph

This repository includes two distinct projects utilizing Neo4j and Python: a Neo4j Resume Graph and a Neo4j Social Network Application. The first project transforms a resume into a Neo4j knowledge graph, while the second develops a social network application with CLI interactions.

## Project Overview

This repository features two projects:

#### Neo4j Resume Graph: 
Converts resume data into a Neo4j graph database to visualize professional profiles.

## Neo4j Resume Graph

Objective: Transform resume data into a Neo4j knowledge graph to visualize and query professional information.

## Overview

In this project, we use Neo4j to model and query professional profiles. The code connects to a Neo4j database and runs Cypher queries to extract skills, education, and experience for a specific person.

## Key Features:

#### Create Nodes:

Created nodes for different entities such as Person, Skill, Education, and Experience.

#### Define Relationships:

Established relationships like HAS_SKILL, STUDIED_AT, and WORKED_AT to link nodes.

#### Run Queries:

Skills Query: Retrieve skills associated with a person.
Education Query: Fetch educational details such as institution name, course, and completion date.
Experience Query: List professional experiences including company, position, duration, and location.

# Development  

#### Connect to Neo4j:
Use the neo4j Python package to interact with the database.
Implement functions for managing users, friendships, and interactions.


## Python Implementation

Neo4j Package: Use the neo4j package for database interactions.
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
