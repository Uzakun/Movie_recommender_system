# 🎬 Movie Recommendation System

A content-based movie recommendation system built with Python and Streamlit that suggests similar movies based on genre analysis using the MovieLens dataset.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.25.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## ✨ Features

- **Content-Based Filtering**: Recommends movies based on genre similarity
- **Interactive Web Interface**: Clean and intuitive Streamlit UI
- **Automatic Dataset Download**: No manual setup required
- **Real-time Recommendations**: Get instant suggestions for any movie
- **Genre Analysis**: Visual representation of genre distribution
- **Error Handling**: Gracefully handles edge cases and missing data

## 🚀 Demo

The application provides a simple interface where users can:
1. Select a movie from the dropdown menu
2. Click "Get Recommendations."
3. View 5 similar movies based on genre analysis

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

## 🎯 Usage

1. **Run the Streamlit app**
```bash
streamlit run movie_recommender.py
```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in your terminal

3. **Use the application**
   - Select a movie from the dropdown menu
   - Click "Get Recommendations."
   - View your personalized recommendations!


## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
- **Requests**: HTTP library for dataset download

### Dependencies

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
streamlit==1.25.0
requests==2.31.0
```

## 📊 Dataset

This project uses the **MovieLens Small Dataset**, which includes:
- 9,742 movies
- 100,836 ratings
- 610 users
- Movie genres and metadata

The dataset is automatically downloaded from the official GroupLens repository on first run.

## 📸 Screenshots

### Main Interface
![Screenshot (112)](https://github.com/user-attachments/assets/a06ea4f6-7ba2-41af-89c9-9eb7cb6d237a)
![Screenshot (110)](https://github.com/user-attachments/assets/98628749-9776-43be-867d-8ede8219d4b5)


### Dataset Statistics
![Screenshot (108)](https://github.com/user-attachments/assets/ba6344a6-f0d7-4612-9b9a-09bbeaa4c053)
![Screenshot (113)](https://github.com/user-attachments/assets/79c6a472-b515-4e57-b94e-b9ae63a8e7a0)


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Future Enhancements

- [ ] Add collaborative filtering option
- [ ] Implement hybrid recommendation system
- [ ] Add movie posters and descriptions
- [ ] Include user authentication
- [ ] Add movie ratings visualization
- [ ] Implement search functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GroupLens](https://grouplens.org/) for providing the MovieLens dataset
- [Streamlit](https://streamlit.io/) for the amazing web framework
- The open-source community for inspiration and resources

---

<p align="center">Made with ❤️ by [Your Name]</p>

<p align="center">
  <a href="https://github.com/yourusername/movie-recommendation-system/issues">Report Bug</a>
  ·
  <a href="https://github.com/yourusername/movie-recommendation-system/issues">Request Feature</a>
</p>
