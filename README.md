# 🧙‍♂️ WordWizard - AI Text Prediction Engine

An advanced Django-based web application that uses deep learning to predict and complete text using TensorFlow/Keras. This project implements a sophisticated LSTM neural network for natural language processing and text generation.

![WordWizard Demo](https://img.shields.io/badge/Django-4.2+-green.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🤖 AI-Powered Text Prediction**: Uses a trained LSTM model to predict the next 12 words
- **🎨 Modern UI**: Beautiful, responsive interface with gradient backgrounds and smooth animations
- **⚡ Fast Predictions**: Optimized model loading with Django caching
- **🔒 Secure**: Updated dependencies with security patches
- **📱 Mobile Friendly**: Responsive design that works on all devices
- **🛡️ Error Handling**: Robust error handling and logging
- **🔧 Production Ready**: Configured for deployment with proper settings

## 🔧 Bug Fixes in This Version

### Critical Bugs Fixed:
1. **🐛 Array Comparison Bug**: Fixed prediction logic where array was being compared to scalar
2. **📁 Path Issues**: Replaced hardcoded relative paths with absolute paths using Django settings
3. **⚠️ Error Handling**: Added comprehensive exception handling and logging
4. **🔒 Security Updates**: Updated all dependencies to latest secure versions
5. **🎯 Input Validation**: Added proper form validation and CSRF protection
6. **💾 Model Loading**: Improved model caching and file existence checks

### UI/UX Improvements:
- Complete UI redesign with modern aesthetics
- Added loading states and user feedback
- Improved accessibility and mobile responsiveness
- Better error messages and user guidance
- Added comprehensive instructions and examples

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/WordWizard-Fixed.git
cd WordWizard-Fixed
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
cd mysite
pip install -r requirements.txt
```

4. **Run migrations**
```bash
python manage.py migrate
```

5. **Start the development server**
```bash
python manage.py runserver
```

6. **Open your browser**
Navigate to `http://127.0.0.1:8000/` to access WordWizard

## 📖 Usage

1. **Enter Text**: Type or paste your text in the input field
2. **Get Predictions**: Click "Predict Next Words" to generate completions
3. **View Results**: The AI will predict the next 12 words to complete your text

### Example Input:
```
"The weather today is beautiful and sunny, perfect for a walk in the park with friends and"
```

### Example Output:
```
"family members who enjoy spending time outdoors during the warm summer months ahead"
```

## 🏗️ Architecture

```
mysite/
├── mysite/              # Django project settings
│   ├── settings.py      # Configuration (updated with security)
│   ├── urls.py          # URL routing
│   └── wsgi.py          # WSGI configuration
├── polls/               # Main application
│   ├── views.py         # View logic (fixed bugs)
│   ├── sustain.py       # Model loading (improved)
│   ├── templates/       # HTML templates (redesigned)
│   ├── nextWord.h5      # Trained LSTM model
│   └── tokenizer.pkl    # Text tokenizer
├── requirements.txt     # Dependencies (updated)
└── manage.py           # Django management
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the `mysite` directory:

```env
Django_Key=your-secret-key-here
DEBUG=True
DNS_address=your-domain.com
IP_address=your-ip-address
```

### Production Settings
For production deployment:
- Set `DEBUG=False`
- Configure proper `ALLOWED_HOSTS`
- Use PostgreSQL database
- Enable security headers
- Configure static files serving

## 📊 Model Information

- **Architecture**: LSTM Neural Network
- **Framework**: TensorFlow/Keras
- **Input Length**: 50 tokens
- **Output**: 12 predicted words
- **Training Data**: Large corpus of English text
- **Model Size**: ~4.6MB

## 🔒 Security Features

- CSRF protection enabled
- Secure headers in production
- Input validation and sanitization
- Updated dependencies with security patches
- Proper error handling without information leakage

## 🧪 Testing

Run the Django test suite:
```bash
python manage.py test
```

For additional testing:
```bash
pip install pytest pytest-django
pytest
```

## 📈 Performance

- **Model Loading**: Cached for faster subsequent requests
- **Response Time**: < 2 seconds for predictions
- **Memory Usage**: Optimized for production deployment
- **Scalability**: Supports horizontal scaling with proper caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues:

**Model file not found**
- Ensure `nextWord.h5` and `tokenizer.pkl` are in the `polls/` directory
- Check file permissions

**Import errors**
- Activate virtual environment
- Install all requirements: `pip install -r requirements.txt`

**CSRF errors**
- Ensure `{% csrf_token %}` is in forms
- Check CSRF middleware is enabled

**Slow predictions**
- Check model caching configuration
- Monitor system memory usage

## 📞 Support

- Create an issue on GitHub for bugs
- Check existing issues for solutions
- Contribute to documentation improvements

## 🎯 Roadmap

- [ ] Add support for different model architectures
- [ ] Implement real-time text completion
- [ ] Add API endpoints for programmatic access
- [ ] Support for multiple languages
- [ ] Advanced text preprocessing options
- [ ] Model fine-tuning interface

---

**🌟 Star this repository if you find it useful!**

Made with ❤️ by [Your Name] | Fixed and Enhanced Version