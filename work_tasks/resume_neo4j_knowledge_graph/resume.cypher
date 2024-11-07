    CREATE
    (amir:Person {name: "Amir Ahmedin", email: "shuaibahmedin@gmail.com", website: "amirahmedin.com"}),

    (aait:Education {name: "AAiT", institution: "AAU", location: "Addis Ababa", GPA: 3.5}),
    (harvard:Education {name: "Harvard University", course: "Python Programming", completion_date: "March 2022", mode: "Online"}),
    (coursera:Education {name: "Coursera", course: "Deep Learning & Tensorflow", completion_date: "February 2024", mode: "Online"}),

    (amir)-[:STUDIED_AT]->(aait),
    (amir)-[:STUDIED_AT]->(harvard),
    (amir)-[:STUDIED_AT]->(coursera),

    (codsoft:Experience {company: "Codsoft", position: "Python Programming Intern", duration: "october 2023", location: "Virtual"}),
    (ai_institute:Experience {company: "Ethiopian AI Institute", position: "AI Intern", duration: "March 2024", location: "Addis Ababa"}),
    (iCog_Labs:Experience {company: "iCog-Labs", position: "AI Intern", duration: "september 2024", location: "Addis Ababa"}),

    (amir)-[:WORKED_AT]->(codsoft),
    (amir)-[:WORKED_AT]->(ai_institute),
    (amir)-[:WORKED_AT]->(iCog_Labs),


    (dental_clinic:Project {name: "Dr. Abdi Speciality Dental Clinic", position: "Web Developer", location: "Addis Ababa"}),
    (a2sv_hackathon:Project {name: "A2SV Internal Hackathon", position: "Backend Developer", location: "Addis Ababa"}),


    (amir)-[:HAVE_DONE]->(dental_clinic),
    (amir)-[:HAVE_DONE]->(a2sv_hackathon),


    (python:Skill {name: "Python", lines_of_code: 2000}),
    (java:Skill {name: "Java", lines_of_code: 500}),
    (js:Skill {name: "JavaScript", lines_of_code: 500}),
    (mysql:Skill {name: "MySQL", familiarity: "Familiar"}),

    (amir)-[:HAS_SKILL]->(python),
    (amir)-[:HAS_SKILL]->(java),
    (amir)-[:HAS_SKILL]->(js),
    (amir)-[:HAS_SKILL]->(mysql)

WITH amir
MATCH (amir:Person {name: "Amir Ahmedin"})-[:STUDIED_AT]->(education:Education {name: "AAiT"}),
    (amir)-[:HAS_SKILL]->(skill:Skill)
WHERE skill.name <> "Python"

WITH skill
MATCH(amir)-[:STUDIED_AT]->(education2:Education)
WHERE education2.name IN ["Harvard University", "Coursera"] AND skill.name = "Python"
RETURN DISTINCT skill.name AS Skill, education2.name AS EducationInstitution
UNION
MATCH (amir:Person {name: "Amir Ahmedin"})-[:STUDIED_AT]->(education:Education {name: "AAiT"}),
      (amir)-[:HAS_SKILL]->(skill:Skill)
// WHERE skill.name <> "Python"
RETURN skill.name AS Skill, education.name AS EducationInstitution;

MATCH (amir)-[:HAS_SKILL]->(skill:Skill)    
RETURN skill.name, skill.lines_of_code; 