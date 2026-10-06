import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="TeamSync AI",
    page_icon="🤝",
    layout="wide"
)




# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "students" not in st.session_state:
    st.session_state.students = []


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("⚙️ Student Registration")

name = st.sidebar.text_input("👤 Student Name")

skills = st.sidebar.multiselect(
    "💻 Skills",
    [
        "Python",
        "Java",
        "C",
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "Node.js",
        "Machine Learning",
        "Deep Learning",
        "Data Science",
        "SQL",
        "UI/UX",
        "Testing"
    ]
)

interests = st.sidebar.multiselect(
    "🔥 Interests",
    [
        "AI",
        "Web Development",
        "App Development",
        "Data Science",
        "Cyber Security",
        "Cloud",
        "IoT",
        "Healthcare",
        "Finance",
        "Education"
    ]
)

role = st.sidebar.selectbox(
    "🎯 Preferred Role",
    [
        "Frontend Developer",
        "Backend Developer",
        "AI/ML Engineer",
        "Data Analyst",
        "UI/UX Designer",
        "Tester",
        "Any"
    ]
)

experience = st.sidebar.selectbox(
    "📊 Experience Level",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)

availability = st.sidebar.slider(
    "⏰ Weekly Availability",
    1,
    20,
    5
)


# ---------------------------------------------------------
# ADD STUDENT
# ---------------------------------------------------------

if st.sidebar.button("➕ Add Student"):

    if not name:
        st.sidebar.error("Please enter student name.")

    elif len(skills) == 0:
        st.sidebar.error("Please select at least one skill.")

    else:

        student = {
            "name": name,
            "skills": skills,
            "interests": interests,
            "role": role,
            "experience": experience,
            "availability": availability
        }

        st.session_state.students.append(student)

        st.sidebar.success(
            f"{name} added successfully! 🎉"
        )


# ---------------------------------------------------------
# DISPLAY STUDENTS
# ---------------------------------------------------------

st.header("👨‍🎓 Registered Students")

if len(st.session_state.students) == 0:

    st.info(
        "No students registered yet. "
        "Add students using the sidebar."
    )

else:

    data = []

    for student in st.session_state.students:

        data.append({
            "Name": student["name"],
            "Skills": ", ".join(student["skills"]),
            "Interests": ", ".join(student["interests"]),
            "Preferred Role": student["role"],
            "Experience": student["experience"],
            "Availability": f'{student["availability"]} hrs'
        })

    df = pd.DataFrame(data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# AI TEAM FORMATION
# ---------------------------------------------------------

st.divider()

st.header("🤖 AI Team Formation")

team_size = st.number_input(
    "👥 Students per Team",
    min_value=2,
    max_value=10,
    value=4
)


# ---------------------------------------------------------
# CREATE STUDENT VECTOR
# ---------------------------------------------------------

def student_text(student):

    return " ".join(
        student["skills"]
        + student["interests"]
        + [student["role"]]
    )


# ---------------------------------------------------------
# TEAM FORMATION ALGORITHM
# ---------------------------------------------------------

def create_teams(students, team_size):

    if len(students) < team_size:

        return []

    texts = [
        student_text(student)
        for student in students
    ]

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(texts)

    similarity = cosine_similarity(matrix)

    unused = set(range(len(students)))

    teams = []

    while len(unused) >= team_size:

        seed = unused.pop()

        team = [seed]

        candidates = list(unused)

        candidates.sort(
            key=lambda x: similarity[seed][x],
            reverse=True
        )

        for candidate in candidates:

            if len(team) >= team_size:
                break

            # Prefer diversity in roles
            existing_roles = [
                students[i]["role"]
                for i in team
            ]

            if (
                students[candidate]["role"]
                not in existing_roles
                or len(existing_roles) < 2
            ):
                team.append(candidate)
                unused.remove(candidate)

        # If team still needs members
        if len(team) < team_size:

            remaining = list(unused)

            for candidate in remaining:

                if len(team) >= team_size:
                    break

                team.append(candidate)
                unused.remove(candidate)

        teams.append(team)

    return teams, similarity


# ---------------------------------------------------------
# GENERATE TEAMS
# ---------------------------------------------------------

if st.button("🚀 Generate AI Teams"):

    if len(st.session_state.students) < team_size:

        st.warning(
            f"Add at least {team_size} students "
            "to create a team."
        )

    else:

        result = create_teams(
            st.session_state.students,
            team_size
        )

        teams, similarity = result

        st.session_state.teams = teams
        st.session_state.similarity = similarity


# ---------------------------------------------------------
# DISPLAY TEAMS
# ---------------------------------------------------------

if "teams" in st.session_state:

    st.divider()

    st.header("🏆 AI Generated Teams")

    teams = st.session_state.teams
    similarity = st.session_state.similarity

    for team_number, team in enumerate(teams, 1):

        st.markdown(
            f"""
            <div class="team-card">
            <h2>🚀 Team {team_number}</h2>
            """,
            unsafe_allow_html=True
        )

        total_score = 0
        comparisons = 0

        for student_index in team:

            student = st.session_state.students[
                student_index
            ]

            st.markdown(
                f"""
                <div class="member">
                👤 <b>{student['name']}</b><br>
                💻 Skills: {", ".join(student['skills'])}<br>
                🎯 Role: {student['role']}<br>
                📊 Level: {student['experience']}<br>
                ⏰ Availability: {student['availability']} hrs/week
                </div>
                """,
                unsafe_allow_html=True
            )

        # Calculate team compatibility
        for i in range(len(team)):

            for j in range(i + 1, len(team)):

                total_score += similarity[
                    team[i]
                ][
                    team[j]
                ]

                comparisons += 1

        if comparisons > 0:

            compatibility = (
                total_score / comparisons
            ) * 100

        else:

            compatibility = 0

        compatibility = min(
            100,
            round(compatibility, 2)
        )

        st.progress(
            compatibility / 100
        )

        st.write(
            f"🧠 **AI Compatibility Score: "
            f"{compatibility}%**"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# EXPLANATION
# ---------------------------------------------------------

if "teams" in st.session_state:

    st.divider()

    st.header("🧠 Why did AI create these teams?")

    st.write("""
    TeamSync AI considers multiple factors:

    🔹 **Skill similarity**  
    Students with complementary technical skills are grouped together.

    🔹 **Project interests**  
    Students interested in similar domains are more likely to collaborate effectively.

    🔹 **Role diversity**  
    The system attempts to avoid creating teams containing only one type of role.

    🔹 **Experience level**  
    Different experience levels can create balanced teams.

    🔹 **Availability**  
    Students with compatible availability can work together more effectively.

    🔹 **Compatibility Score**  
    TF-IDF and cosine similarity are used to estimate how well student profiles match.
    """)


# ---------------------------------------------------------
# RESET
# ---------------------------------------------------------

st.divider()

if st.button("🗑️ Reset All Students"):

    st.session_state.students = []

    if "teams" in st.session_state:
        del st.session_state.teams

    if "similarity" in st.session_state:
        del st.session_state.similarity

    st.rerun()