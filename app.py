from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# هذا الراوت للصفحة الرئيسية
@app.route('/')
def home():
    return render_template('index.html')

# هذا الراوت اللي يستقبل طلب الـ AI (سنقوم بتفعيله لاحقاً)
@app.route('/process', methods=['POST'])
def process_ai():
    data = request.form.get('user_input')
    # هنا بنحط كود استدعاء المودل بعدين
    result = f"تم استلام النص: {data} (هذا رد تجريبي)"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)