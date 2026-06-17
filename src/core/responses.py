"""
Local keyword-based intent detector and dynamic response generator.

Before calling the LLM for intent classification, check if the user message
matches a known keyword. If it does, return a native response immediately.
"""

import re
import random
import unicodedata

# ===========================================================================
# 1. THE KEYWORD DICTIONARY (For Intent Detection)
# ===========================================================================

INTENT_KEYWORDS: dict[str, list[str]] = {
    "greeting": [
        # English
        "hi", "hey", "hello", "howdy", "hiya", "sup", "what's up", "whats up",
        "good morning", "good afternoon", "good evening", "good day", "greetings",
        "how are you", "how r u", "how are u", "how do you do", "how's it going",
        "hows it going", "how are things", "what's good", "yo",
        # Arabic
        "مرحبا", "مرحباً", "أهلا", "أهلاً", "السلام عليكم", "صباح الخير",
        "مساء الخير", "كيف حالك", "كيف الحال", "هاي", "هلا",
        # Bulgarian
        "здравей", "здравейте", "привет", "добро утро", "добър ден",
        "добър вечер", "как си", "как сте",
        # German
        "hallo", "guten morgen", "guten tag", "guten abend", "wie geht es",
        "wie geht's", "wie gehts", "servus", "moin", "grüß gott",
        # Greek
        "γεια", "γεια σου", "γεια σας", "καλημέρα", "καλησπέρα",
        "καλό απόγευμα", "τι κάνεις", "τι κάνετε",
        # Spanish
        "hola", "buenos días", "buenas tardes", "buenas noches", "buenas",
        "qué tal", "cómo estás", "cómo está", "cómo andas",
        # French
        "bonjour", "bonsoir", "salut", "coucou", "comment ça va",
        "comment allez-vous", "comment vas-tu", "ça va",
        # Hindi
        "नमस्ते", "नमस्कार", "हेलो", "हाय", "कैसे हो", "कैसे हैं",
        "सुप्रभात", "शुभ संध्या",
        # Italian
        "ciao", "salve", "buongiorno", "buonasera", "come stai",
        "come sta", "come va",
        # Japanese
        "こんにちは", "おはよう", "おはようございます", "こんばんは",
        "やあ", "はじめまして", "お元気ですか",
        # Dutch
        "hoi", "dag", "goedemorgen", "goedemiddag", "goedenavond",
        "hoe gaat het", "hoe is het",
        # Polish
        "cześć", "hej", "dzień dobry", "dobry wieczór", "jak się masz",
        "jak się czujesz", "siema", "witaj",
        # Portuguese
        "olá", "oi", "bom dia", "boa tarde", "boa noite",
        "como vai", "como está", "como estás", "tudo bem",
        # Russian
        "привет", "здравствуй", "здравствуйте", "добрый день",
        "доброе утро", "добрый вечер", "как дела", "как ты",
        # Swahili
        "habari", "hujambo", "jambo", "mambo", "salama", "shikamoo",
        "habari yako", "habari za asubuhi",
        # Thai
        "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "หวัดดี", "เป็นยังไงบ้าง",
        # Turkish
        "merhaba", "selam", "günaydın", "iyi günler", "iyi akşamlar",
        "nasılsın", "nasılsınız", "naber",
        # Urdu
        "ہیلو", "ہائے", "السلام علیکم", "آداب", "صبح بخیر",
        "شام بخیر", "کیسے ہیں", "کیا حال ہے",
        # Vietnamese
        "xin chào", "chào", "chào bạn", "chào anh", "chào chị",
        "bạn khỏe không", "khỏe không",
        # Chinese
        "你好", "您好", "早上好", "下午好", "晚上好",
        "嗨", "哈喽", "你好吗", "最近怎么样",
    ],

    "goodbye": [
        # English
        "bye", "goodbye", "good bye", "see you", "see ya", "later",
        "take care", "farewell", "until next time", "talk later",
        "gotta go", "i'm leaving", "im leaving", "signing off", "cya",
        "ttyl", "ttys", "good night", "goodnight", "have a good day",
        # Arabic
        "وداعاً", "وداعا", "إلى اللقاء", "مع السلامة", "باي",
        "تصبح على خير", "مع السلام",
        # Bulgarian
        "довиждане", "чао", "па", "лека нощ", "до скоро",
        # German
        "tschüss", "auf wiedersehen", "ciao", "bis später", "bis bald",
        "gute nacht", "mach's gut", "servus",
        # Greek
        "αντίο", "γεια", "γεια χαρά", "τα λέμε", "καληνύχτα",
        # Spanish
        "adiós", "adios", "hasta luego", "chao", "nos vemos",
        "buenas noches", "hasta pronto", "cuídate",
        # French
        "au revoir", "adieu", "à bientôt", "bonne nuit", "salut",
        "à tout à l'heure", "bonne journée",
        # Hindi
        "अलविदा", "बाय", "फिर मिलेंगे", "शुभ रात्रि", "जाता हूँ",
        # Italian
        "arrivederci", "addio", "a presto", "buonanotte", "ci vediamo",
        # Japanese
        "さようなら", "バイバイ", "じゃあね", "またね", "おやすみ",
        "おやすみなさい", "失礼します",
        # Dutch
        "doei", "dag", "tot ziens", "tot later", "goedenacht",
        # Polish
        "do widzenia", "pa", "na razie", "dobranoc", "do zobaczenia",
        # Portuguese
        "tchau", "adeus", "até logo", "boa noite", "até amanhã",
        # Russian
        "пока", "до свидания", "до встречи", "спокойной ночи",
        "увидимся",
        # Swahili
        "kwaheri", "kwa heri", "lala salama", "tutaonana",
        # Thai
        "ลาก่อน", "บาย", "คืนนี้ฝันดี", "แล้วเจอกัน",
        # Turkish
        "hoşça kal", "güle güle", "görüşürüz", "iyi geceler", "bay bay",
        # Urdu
        "الوداع", "خدا حافظ", "پھر ملیں گے", "شب بخیر", "بائے",
        # Vietnamese
        "tạm biệt", "bái bai", "hẹn gặp lại", "chúc ngủ ngon",
        # Chinese
        "再见", "拜拜", "晚安", "回头见", "保重",
    ],

    "gratitude": [
        # English
        "thank you", "thanks", "thank u", "thx", "ty", "cheers",
        "much appreciated", "i appreciate it", "appreciate that",
        "that's helpful", "that was helpful", "helpful", "you're great",
        "you are great", "you're amazing", "that helped",
        # Arabic
        "شكراً", "شكرا", "جزاك الله خيراً", "ممنون", "مشكور",
        "شكراً جزيلاً",
        # Bulgarian
        "благодаря", "мерси", "много благодаря",
        # German
        "danke", "danke schön", "vielen dank", "herzlichen dank",
        "dankeschön",
        # Greek
        "ευχαριστώ", "ευχαριστώ πολύ", "σ'ευχαριστώ",
        # Spanish
        "gracias", "muchas gracias", "te lo agradezco",
        "se lo agradezco",
        # French
        "merci", "merci beaucoup", "je vous remercie",
        # Hindi
        "धन्यवाद", "शुक्रिया", "बहुत शुक्रिया", "आभार",
        # Italian
        "grazie", "grazie mille", "ti ringrazio",
        # Japanese
        "ありがとう", "ありがとうございます", "どうもありがとう",
        "助かりました",
        # Dutch
        "bedankt", "dank je", "dank u", "dank je wel", "dank u wel",
        # Polish
        "dziękuję", "dzięki", "bardzo dziękuję",
        # Portuguese
        "obrigado", "obrigada", "muito obrigado", "muito obrigada",
        # Russian
        "спасибо", "большое спасибо", "благодарю",
        # Swahili
        "asante", "asante sana", "nashukuru",
        # Thai
        "ขอบคุณ", "ขอบคุณมาก", "ขอบคุณครับ", "ขอบคุณค่ะ",
        # Turkish
        "teşekkürler", "teşekkür ederim", "çok teşekkürler",
        # Urdu
        "شکریہ", "بہت شکریہ", "ممنون", "مہربانی",
        # Vietnamese
        "cảm ơn", "cảm ơn bạn", "cảm ơn nhiều",
        # Chinese
        "谢谢", "谢谢你", "非常感谢", "多谢",
    ],
}


# ===========================================================================
# 2. THE MULTI-LINGUAL RESPONSE DICTIONARY (For Final Output)
# ===========================================================================

MULTI_LINGUAL_RESPONSES = {
    "en": {
        "greeting": [
            "Hello! I am here to listen. How are you doing today?",
            "Hi there. How are you feeling right now?",
            "Greetings. What's on your mind today?"
        ],
        "goodbye": [
            "Take care. Remember, there is always support available if you need it.",
            "Goodbye for now. Stay safe.",
            "Talk to you later. I'll be right here whenever you need someone to listen."
        ],
        "gratitude": [
            "You are very welcome. I am glad I could help.",
            "Anytime. I'm here to support you.",
            "You're welcome. It takes courage to reach out."
        ],
        "out_of_scope": [
            "I am a mental health assistant. I cannot help with coding, general trivia, or physical medical advice. How can I support your mental well-being today?"
        ]
    },
    "ar": {
        "greeting": [
            "مرحباً! أنا هنا للاستماع إليك. كيف حالك اليوم؟",
            "أهلاً بك. بماذا تشعر الآن؟",
            "السلام عليكم. هل هناك ما تود التحدث عنه؟"
        ],
        "goodbye": [
            "اعتنِ بنفسك. تذكر أن الدعم متاح دائماً إذا احتجت إليه.",
            "مع السلامة. أتمنى لك يوماً هادئاً.",
            "إلى اللقاء. سأكون هنا متى أردت التحدث."
        ],
        "gratitude": [
            "على الرحب والسعة. يسعدني أنني تمكنت من المساعدة.",
            "لا شكر على واجب. أنا هنا لدعمك دائماً.",
            "العفو. التواصل وطلب الدعم خطوة شجاعة منك."
        ],
        "out_of_scope": [
            "أنا مساعد متخصص في الصحة النفسية. لا يمكنني المساعدة في البرمجة أو المعلومات العامة. كيف يمكنني دعم صحتك النفسية اليوم؟"
        ]
    },
    "bg": {
        "greeting": [
            "Здравей! Тук съм, за да те изслушам. Как си днес?",
            "Привет. Как се чувстваш в този момент?"
        ],
        "goodbye": [
            "Пази се. Помни, че винаги има подкрепа, ако имаш нужда.",
            "Довиждане за сега. Бъди в безопасност."
        ],
        "gratitude": [
            "Моля, за мен е удоволствие. Радвам се, че помогнах.",
            "Няма защо. Тук съм, за да те подкрепя."
        ],
        "out_of_scope": [
            "Аз съм асистент за психично здраве. Не мога да помагам с програмиране или общи въпроси. Как мога да подкрепя емоционалното ти състояние днес?"
        ]
    },
    "de": {
        "greeting": [
            "Hallo! Ich bin hier, um zuzuhören. Wie geht es dir heute?",
            "Hallo. Wie fühlst du dich im Moment?",
            "Grüß dich. Was beschäftigt dich heute?"
        ],
        "goodbye": [
            "Pass gut auf dich auf. Denk daran, dass immer Unterstützung da ist, wenn du sie brauchst.",
            "Auf Wiedersehen. Bleib sicher.",
            "Bis bald. Ich bin da, wenn du jemanden zum Zuhören brauchst."
        ],
        "gratitude": [
            "Sehr gerne. Ich bin froh, dass ich helfen konnte.",
            "Jederzeit. Ich bin hier, um dich zu unterstützen.",
            "Gern geschehen. Es erfordert Mut, sich Hilfe zu suchen."
        ],
        "out_of_scope": [
            "Ich bin ein Assistent für mentale Gesundheit. Ich kann nicht bei Programmierfragen oder allgemeinem Wissen helfen. Wie kann ich heute dein seelisches Wohlbefinden unterstützen?"
        ]
    },
    "el": {
        "greeting": [
            "Γεια σας! Είμαι εδώ για να σας ακούσω. Πώς είστε σήμερα;",
            "Γεια σου. Πώς νιώθεις αυτή τη στιγμή;"
        ],
        "goodbye": [
            "Να προσέχετε τον εαυτό σας. Να θυμάστε ότι υπάρχει πάντα υποστήριξη αν τη χρειαστείτε.",
            "Εις το επανιδείν. Να είστε ασφαλείς."
        ],
        "gratitude": [
            "Παρακαλώ, χαρά μου. Χαίρομαι που μπόρεσα να βοηθήσω.",
            "Δεν κάνει τίποτα. Είμαι εδώ για να σε υποστηρίξω."
        ],
        "out_of_scope": [
            "Είμαι βοηθός ψυχικής υγείας. Δεν μπορώ να βοηθήσω με προγραμματισμό ή γενικές ερωτήσεις. Πώς μπορώ να υποστηρίξω την ψυχική σου ευεξία σήμερα;"
        ]
    },
    "es": {
        "greeting": [
            "¡Hola! Estoy aquí para escucharte. ¿Cómo te sientes hoy?",
            "Saludos. ¿Qué tienes en mente hoy?"
        ],
        "goodbye": [
            "Cuídate mucho. Recuerda que siempre hay apoyo disponible si lo necesitas.",
            "Adiós por ahora. Estaré aquí cuando necesites hablar."
        ],
        "gratitude": [
            "De nada. Me alegra haber podido ayudar.",
            "No hay de qué. Estoy aquí para apoyarte.",
            "Un placer. Se necesita valor para buscar ayuda."
        ],
        "out_of_scope": [
            "Soy un asistente de salud mental. No puedo ayudar con programación, preguntas generales o consejos médicos. ¿Cómo puedo apoyar tu bienestar emocional hoy?"
        ]
    },
    "fr": {
        "greeting": [
            "Bonjour ! Je suis là pour vous écouter. Comment allez-vous aujourd'hui ?",
            "Salut. Comment vous sentez-vous en ce moment ?"
        ],
        "goodbye": [
            "Prenez soin de vous. N'oubliez pas que de l'aide est toujours disponible.",
            "Au revoir. Passez une journée paisible.",
            "À bientôt. Je serai là dès que vous aurez besoin d'une oreille attentive."
        ],
        "gratitude": [
            "Je vous en prie. Je suis heureux d'avoir pu vous aider.",
            "Avec plaisir. Je suis là pour vous soutenir.",
            "De rien. C'est courageux de faire le premier pas."
        ],
        "out_of_scope": [
            "Je suis un assistant en santé mentale. Je ne peux pas répondre aux questions de code ou de culture générale. Comment puis-je vous soutenir mentalement aujourd'hui ?"
        ]
    },
    "hi": {
        "greeting": [
            "नमस्ते! मैं यहाँ आपकी बात सुनने के लिए हूँ। आज आप कैसा महसूस कर रहे हैं?",
            "हेलो। आपके मन में आज क्या चल रहा है?"
        ],
        "goodbye": [
            "अपना ख्याल रखें। याद रखें, ज़रूरत पड़ने पर हमेशा मदद उपलब्ध है।",
            "अलविदा। सुरक्षित रहें।"
        ],
        "gratitude": [
            "आपका स्वागत है। मुझे खुशी है कि मैं मदद कर सका।",
            "कोई बात नहीं। मैं यहाँ आपका साथ देने के लिए हूँ।"
        ],
        "out_of_scope": [
            "मैं एक मानसिक स्वास्थ्य सहायक हूँ। मैं कोडिंग या सामान्य ज्ञान में मदद नहीं कर सकता। आज मैं आपके मानसिक स्वास्थ्य को बेहतर बनाने में कैसे मदद करूँ?"
        ]
    },
    "it": {
        "greeting": [
            "Ciao! Sono qui per ascoltarti. Come stai oggi?",
            "Salve. Come ti senti in questo momento?"
        ],
        "goodbye": [
            "Prenditi cura di te. Ricorda wild che il supporto è sempre disponibile se ne hai bisogno.",
            "Arrivederci per ora. Ti auguro una giornata tranquilla."
        ],
        "gratitude": [
            "Prego, figurati. Sono felice di essere stato d'aiuto.",
            "Di nulla. Sono qui per sostenerti."
        ],
        "out_of_scope": [
            "Sono un assistente per la salute mentale. Non posso aiutarti con la programmazione o domande generali. Come posso supportare il tuo benessere emotivo oggi?"
        ]
    },
    "ja": {
        "greeting": [
            "こんにちは。お話を聞かせてください。今日のご気分はいかがですか？",
            "何か心に引っかかっていることはありますか？"
        ],
        "goodbye": [
            "お体に気をつけて。必要なときは、らいつでもサポートがあることを忘れないでくださいね。",
            "それではまた。穏やかな一日をお過ごしください。"
        ],
        "gratitude": [
            "どういたしまして。お役に立てて嬉しいです。",
            "いつでもどうぞ。私はあなたをサポートするためにここにいます。"
        ],
        "out_of_scope": [
            "私はメンタルヘルスのアシスタントです。プログラミングや一般的な質問にはお答えできません。今日はどのような心のサポートが必要ですか？"
        ]
    },
    "nl": {
        "greeting": [
            "Hallo! Ik ben er om naar je te luisteren. Hoe gaat het vandaag met je?",
            "Hoi. Hoe voel je je op dit moment?"
        ],
        "goodbye": [
            "Zorg goed voor jezelf. Onthoud dat er altijd hulp beschikbaar is als je die nodig hebt.",
            "Tot ziens voor nu. Pas goed op jezelf."
        ],
        "gratitude": [
            "Graag gedaan. Ik ben blij dat ik kon helpen.",
            "Geen dank. Ik ben er om je te ondersteunen."
        ],
        "out_of_scope": [
            "Ik ben een assistent voor mentale gezondheid. Ik kan niet helpen met programmeren of algemene kennis. Hoe kan ik je mentaal welzijn vandaag ondersteunen?"
        ]
    },
    "pl": {
        "greeting": [
            "Cześć! Jestem tutaj, aby Cię wysłuchać. Jak się dzisiaj masz?",
            "Witaj. Jak się czujesz w tej chwili?"
        ],
        "goodbye": [
            "Dbaj o siebie. Pamiętaj, że pomoc jest zawsze dostępna, jeśli jej potrzebujesz.",
            "Do zobaczenia. Trzymaj się ciepło."
        ],
        "gratitude": [
            "Nie ma za co. Cieszę się, że mogłem pomóc.",
            "Cała przyjemność po mojej stronie. Jestem tu, by Cię wspierać."
        ],
        "out_of_scope": [
            "Jestem asystentem ds. zdrowia psychicznego. Nie pomagam w kodowaniu ani nie odpowiadam na pytania ogólne. Jak mogę dziś wesprzeć Twoje samopoczucie?"
        ]
    },
    "pt": {
        "greeting": [
            "Olá! Estou aqui para te escutar. Como você está se sentindo hoje?",
            "Oi. O que está na sua mente hoje?"
        ],
        "goodbye": [
            "Cuide-se bem. Lembre-se de que sempre há apoio disponível se você precisar.",
            "Até logo. Fique bem."
        ],
        "gratitude": [
            "De nada. Fico feliz em poder ajudar.",
            "Disponha. Estou aqui para te apoiar."
        ],
        "out_of_scope": [
            "Sou um assistente de saúde mental. Não posso ajudar com programação ou perguntas gerais. Como posso apoiar seu bem-estar emocional hoje?"
        ]
    },
    "ru": {
        "greeting": [
            "Здравствуйте! Я здесь, чтобы выслушать вас. Как вы себя чувствуете сегодня?",
            "Привет. Что у вас на душе сегодня?"
        ],
        "goodbye": [
            "Берегите себя. Помните, что вы всегда можете получить поддержку, если она понадобится.",
            "До свидания. Всего вам доброго."
        ],
        "gratitude": [
            "Пожалуйста. Я рад, что смог помочь.",
            "Не за что. Я здесь, чтобы поддержать вас."
        ],
        "out_of_scope": [
            "Я ассистент по ментальному здоровью. Я не помогаю с программированием или общими вопросами. Как я могу поддержать ваше эмоциональное состояние сегодня?"
        ]
    },
    "sw": {
        "greeting": [
            "Habari! Nipo hapa kukusikiliza. Habari za leo?",
            "Jambo. Unajisikiaje wakati huu?"
        ],
        "goodbye": [
            "Jitunze. Kumbuka, msaada unapatikana kila wakati ukihitaji.",
            "Kwaheri kwa sasa. Kuwa salama."
        ],
        "gratitude": [
            "Karibu sana. Nafurahi nimeweza kusaidia.",
            "Asante pia. Nipo hapa kukusaidia."
        ],
        "out_of_scope": [
            "Mimi ni msaidizi wa afya ya akili. Siwezi kusaidia na masuala ya programu au maswali ya jumla. Nawezaje kusaidia ustawi wako wa akili leo?"
        ]
    },
    "th": {
        "greeting": [
            "สวัสดีค่ะ/ครับ ฉันอยู่ที่นี่เพื่อรับฟังคุณ วันนี้คุณเป็นอย่างไรบ้าง?",
            "สวัสดี มีอะไรในใจที่อยากเล่าให้ฟังไหม?"
        ],
        "goodbye": [
            "ดูแลตัวเองด้วยนะ จำไว้ว่ายังมีคนที่พร้อมช่วยเหลือคุณเสมอ",
            "ลาก่อน ขอให้วันนี้เป็นวันที่ดีนะ"
        ],
        "gratitude": [
            "ยินดีมากๆ ค่ะ/ครับ ดีใจที่ได้ช่วยนะ",
            "ไม่เป็นไรเลย ฉันพร้อมอยู่เคียงข้างคุณเสมอ"
        ],
        "out_of_scope": [
            "ฉันเป็นผู้ช่วยด้านสุขภาพจิต ไม่สามารถช่วยเหลือเรื่องการเขียนโค้ดหรือคำถามทั่วไปได้ วันนี้มีเรื่องสุขภาพจิตอะไรที่อยากให้ฉันรับฟังไหม?"
        ]
    },
    "tr": {
        "greeting": [
            "Merhaba! Sizi dinlemek için buradayım. Bugün nasıl hissediyorsunuz?",
            "Selam. Bugün aklınızdan neler geçiyor?"
        ],
        "goodbye": [
            "Kendinize çok iyi bakın. İhtiyacınız olduğunda her zaman destek bulabileceğinizi unutmayın.",
            "Şimdilik hoşça kalın. Güvende kalın."
        ],
        "gratitude": [
            "Rica ederim. Yardımcı olabildiğime sevindim.",
            "Her zaman. Sizi desteklemek için buradayım."
        ],
        "out_of_scope": [
            "Ben bir ruh sağlığı asistanıyım. Kodlama veya genel kültür sorularında yardımcı olamam. Bugün ruhsal esenliğinizi desteklemek için ne yapabilirim?"
        ]
    },
    "ur": {
        "greeting": [
            "ہیلو! میں یہاں آپ کی بات سننے کے لیے موجود ہوں۔ آج آپ کا کیا حال ہے؟",
            "السلام علیکم۔ آپ اس وقت کیسا محسوس کر رہے ہیں؟"
        ],
        "goodbye": [
            "اپنا خیال رکھیں۔ یاد رکھیں، ضرورت پڑنے پر مدد ہمیشہ دستیاب ہے۔",
            "خدا حافظ۔ محفوظ رہیں۔"
        ],
        "gratitude": [
            "خوش آمدید۔ مجھے خوشی ہے کہ میں کام آسکا۔",
            "کوئی بات نہیں۔ میں یہاں آپ کی مدد کے لیے ہی ہوں۔"
        ],
        "out_of_scope": [
            "میں ذہنی صحت کا معاون ہوں۔ میں کوڈنگ یا عام معلومات کے سوالات میں مدد نہیں کر سکتا۔ آج میں آپ کی ذہنی صحت کے لیے کیا مدد کر سکتا ہوں؟"
        ]
    },
    "vi": {
        "greeting": [
            "Xin chào! Tôi ở đây để lắng nghe bạn. Hôm nay bạn thế nào?",
            "Chào bạn. Hiện tại bạn đang cảm thấy thế nào?"
        ],
        "goodbye": [
            "Hãy chăm sóc bản thân thật tốt nhé. Hãy nhớ rằng luôn có sự hỗ trợ khi bạn cần.",
            "Tạm biệt nhé. Chúc bạn một ngày bình yên."
        ],
        "gratitude": [
            "Không có gì đâu. Tôi rất vui vì đã giúp được bạn.",
            "Luôn sẵn sàng. Tôi ở đây để đồng hành cùng bạn."
        ],
        "out_of_scope": [
            "Tôi là trợ lý hỗ trợ sức khỏe tinh thần. Tôi không thể giúp lập trình hay trả lời câu hỏi thông thường. Hôm nay tôi có thể hỗ trợ gì cho tâm trạng của bạn?"
        ]
    },
    "zh": {
        "greeting": [
            "你好！我在这里倾听。你今天感觉怎么样？",
            "你好。现在有什么想聊聊的吗？"
        ],
        "goodbye": [
            "请多保重。请记住，需要的时候总会有支持在你身边。",
            "再见。愿你拥有平静的一天。"
        ],
        "gratitude": [
            "不客气。很高兴能帮到你。",
            "随时乐意。我会一直在这里支持你。",
            "不用谢。愿意倾诉就是充满勇气的一步。"
        ],
        "out_of_scope": [
            "我是心理健康助手。我无法提供编程、常识或医疗方面的建议。今天我该如何支持你的心理健康呢？"
        ]
    }
}

# ===========================================================================
# 3. FAST MATCHING AND RESPONSE FUNCTIONS
# ===========================================================================

# Punctuation characters we strip before comparing (covers all scripts)
_STRIP_PUNCT = re.compile(
    r"[\s\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
    r"\u00A1-\u00BF\u060C\u061B\u061F\uFF01-\uFF0F\uFF1A-\uFF20"
    r"\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F\u2018-\u201F]+"
)

def _normalise(text: str) -> str:
    """Lowercase, NFKC-normalise, strip surrounding punctuation & whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _STRIP_PUNCT.sub("", text).strip()

def _is_cjk(text: str) -> bool:
    """Return True if the text contains any CJK character."""
    return any(unicodedata.category(c) in ("Lo",) and ord(c) > 0x2E7F for c in text)

_EXACT: dict[str, set[str]] = {}
_LATIN_MULTI: dict[str, re.Pattern] = {}

for _intent, _kws in INTENT_KEYWORDS.items():
    # Build exact match set
    _EXACT[_intent] = {_normalise(kw) for kw in _kws}

    # Multi-word Latin phrases only (for whole-message regex fallback)
    _latin_phrases = [
        re.escape(kw.lower())
        for kw in _kws
        if " " in kw and not _is_cjk(kw)
    ]
    if _latin_phrases:
        # Anchored full-string match, allowing leading/trailing punctuation
        _LATIN_MULTI[_intent] = re.compile(
            r"^[\s\W]*(" + "|".join(
                sorted(_latin_phrases, key=len, reverse=True)
            ) + r")[\s\W]*$",
            re.IGNORECASE,
        )
    else:
        _LATIN_MULTI[_intent] = None  # type: ignore

def detect_intent_from_keywords(text: str) -> str | None:
    """Return an intent if the message is *entirely* a known keyword, else None.

    Strategy
    --------
    1. Check length (skip LLM boundary check).
    2. Normalise the input (NFKC lowercase, strip punctuation).
    3. Exact-set lookup — works for every script including CJK.
    4. Anchored regex — catches multi-word Latin phrases like "good morning".
    5. Return None if neither matches.
    """
    if not text or not text.strip():
        return None
        
    # Extra safety guard: If the message is very long, it contains context. 
    # Skip keywords and go straight to the LLM.
    if len(text) > 40:
        return None

    normalised = _normalise(text)

    for intent in ("greeting", "gratitude", "goodbye"):
        # --- exact match (single-word, CJK phrase, or space-stripped Latin phrase) ---
        if normalised in _EXACT[intent]:
            return intent

        # --- anchored full-string match for multi-word Latin phrases ---
        pattern = _LATIN_MULTI.get(intent)
        if pattern and pattern.fullmatch(text.strip()):
            return intent

    return None

def get_dynamic_response(intent: str, language_code: str) -> str:
    """Fetches a random, native response for the intent without using the API."""
    lang_dict = MULTI_LINGUAL_RESPONSES.get(language_code, MULTI_LINGUAL_RESPONSES["en"])
    responses = lang_dict.get(intent, lang_dict.get("out_of_scope", ["I am here to help with mental health."]))
    return random.choice(responses)