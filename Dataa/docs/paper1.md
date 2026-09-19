A T5-BASED TRANSFORMER MODEL FOR INTERPRETABLE BIDIRECTIONAL 
TRANSLATION OF INFORMAL GEN-Z LANGUAGE 
Harshit Jaiswal1*, Sumitkumar Tripathi2 and Kanojia Mahendra3 
1
Information Technology, Sheth L.U. Jhaveri and Sir M.V. College, India, bscit.harshitjaiswal@gmail.com 
2
Information Technology, Sheth L.U. Jhaveri and Sir M.V. College, India, sumit11.tripathi@gmail.com 
3Department of Computer Science, Sheth. L.U.J. and Sir M.V. College, India. kgkmahendra@gmail.com 
*Corresponding author: Harshit Jaiswal, bscit.harshitjaiswal@gmail.com 
ABSTRACT 
Developments in natural language processing (NLP) have highlight the growing need to tackle informal digital 
language. However, understanding and translating the rapidly changing Gen-Z slang is difficult. The increase 
in use of slang during online communication has created a language gap between younger users and those less 
familiar with new expressions, especially millennials and non-social media users. This research focuses on a 
two-way translation approach using a fine-tuned Text-to-Text Transfer Transformer (T5) model trained on an 
expanded English–Gen-Z parallel dataset. This proposed framework treats slang translation as a single text 
generation task. This helps the model capture style variations while keeping the original meaning intact, even 
with limitations like limited data and changing language patterns. The findings show that the Transformer
based method improves translation accuracy and clarity compared to traditional rule-based normalization 
methods. The model achieves noticeable gains in translation quality measured by BLEU scores and semantic 
similarity metrics. Lexical mapping analysis also shows consistent slang transformation patterns that make 
understanding easier. These results suggest that fine-tuned Transformer models offer an impactful way to 
process informal language and reduce communication gap between different generations. This approach has 
practical applications in social media communication, language accessibility tools, and NLP systems that need 
to handle dynamic slang expressions. This will help many people who are not familiar with the new fast
growing slangs can now understand and adapt with these slangs 
Keywords: Gen-Z Slang Translation, Transformer-based NLP, T5 Model, Informal Language Processing, Text
to-Text Generation, Bidirectional Translation, Social Media Language. 
1. INTRODUCTION 
Computational Linguistics and Natural Language Processing (NLP) have become important for modern digital 
systems that work with analyzing, understanding, and generating human language. Recent research in deep 
learning, especially in Transformer-based architectures, have improved how NLP models understand sentence 
context, relationships between words, and meaning of the sentence through attention mechanisms (Vaswani et 
al., 2017; Brown et al., 2020). Text-to-text frameworks like Text-to-Text Transfer Transformer (T5), increase 
these capabilities by treating multiple NLP tasks within a single generative structure. This method allows more 
flexible and scalable language processing (Raffel et al., 2020; Yano et al., 2024). Even with this type of 
progress, many NLP systems are still struggling with socially influenced and non-formal forms of language that 
is different from formal grammar. This challenge becomes clearer in online spaces, where communication is 
done by cultural trends, community behavior, and rapid expression change (Eisenstein, 2013; Yang & 
Eisenstein, 2017). One clear example is Gen-Z slang, which mainly grows from social media and digital 
interactions. This form of language evolves quickly, depends heavily on context, and often carries emotional or 
social meaning beyond its literal understanding (Hovy & Yang, 2021). Generation Z commonly referred to as ―Zoomers‖ generally includes individuals born between 1997 and 2012 and is strongly linked with 
communication styles influenced by digital platforms and real-time conversation (Pew Research Center, 2019). 
Previous research showed us that NLP models trained mainly on formal or semi-formal datasets and they 
struggled with different slang, informal text or incomplete forms slangs. They may misunderstand sentiment, 
intent, or practical meaning in such contexts (Eisenstein, 2013; Yang & Eisenstein, 2017). Earlier attempts 
heavily depend on fixed slang dictionaries and rule-based normalization methods, which lack flexibility and 
perform poorly when slang appears in mixed or incomplete forms (Bucur et al., 2021; Felbo et al., 2017). Older 
model lack in understanding the context of the sentence. But, modern Transformer-based models such as BERT 
and RoBERTa have improved contextual understanding and adaptability in many language tasks (Devlin et al., 
2019; Liu et al., 2019), most existing research still see slang handling as a one-way normalization and 
translation problem rather than a bidirectional translation task. This limits their ability in interactive systems and 
real-world conversational moment (Bucur et al., 2021; Camacho-Collados et al., 2022). This results in growing 
communication gap between regular digital users and person who are less familiar with evolving slang 
292 
International Journal of Advance and Innovative Research   
Volume 13, Issue 1 (XIII): January – March 2026 
ISSN 2394 - 7780 
expressions. This gap highlights the need for adaptable translation systems capable of understanding and 
generating slang in changing social contexts. To address this need, this study proposes a bidirectional Gen-Z 
slang translation framework based on a Transformer-driven T5 model supported by controlled lexical mapping. 
This system translates between standard English and Gen-Z slang in both directions while keep the original 
meaning, emotional tone, and conversational intent safe. In addition to this, a lexical explainability component 
provide and help us to understand word-level transformation information, improving transparency and user 
understanding (Clark et al., 2019; Sap et al., 2020; Fantozzi & Naldi, 2024). By combining contextual neural 
modeling with structured lexical guidance, this research tries to reduce the gap between fixed rule-based 
methods and purely neural methods. The objective is to support overall, understandable, and socially aware 
language technologies that can work with evolving communication slangs in digital environments (Camacho
Collados et al., 2022; Ishita & Mamidi, 2025; Zhu et al., 2025). 
2. LITERATURE REVIEW 
Natural Language Processing (NLP) has changed a lot in past few years and this change came from better deep 
learning studies and techniques and more access to large text datasets. These changes improved NLP and 
opened a way for new researchers to learn and develope more advanced and better models. Earlier NLP systems 
heavily depended on static rule-based methods, statistical modeling, and regular machine learning techniques 
that used create features manually. These methods worked well in structured and formal settings. However, they 
struggled with the informal language often found in online communication. The rapid growth of social media 
highlighted these issues since digital language frequently departs from standard grammar and evolves quickly. 
This made it tough for traditional systems to keep up (Eisenstein, 2013). Researchers then began working with 
neural language models, which changed how NLP systems used to design and train. Encoder-decoder 
architectures helped models to learn directly the connections between input and output sequences. Because of 
this shift, tasks like machine translation, paraphrasing, and text normalization started showing great 
improvement (Johnson et al., 2017; Gehring et al., 2017). Despite these advancements, previous neural models 
had many problems to understand relationships between words that were far apart in a sentence. In practice, 
performance dropped when sentences became longer or more complex. The introduction of self-attention-based 
Transformer models has addressed many limitations of earlier neural NLP approaches by using contextual 
understanding across full sequences. The self-attention mechanism enables the model to capture relationships 
between slang tokens and their formal counterparts during translation (Vaswani et al., 2017). In real-world 
applications, this progress perfectly improved performance in tasks such as translation, summarization, and 
language representation. Later developments showed the importance of contextual embeddings and 
understanding the meaning of the sentence. Bidirectional Encoder Representations from Transformers (BERT) 
represents that understanding words using both previous and next context leads to stronger language 
understanding and improved evaluation results (Devlin et al., 2019). Next improvements, such as RoBERTa, 
used better training strategies and larger datasets, this resulted in stronger contextual understanding (Liu et al., 
2019). At the same time, methods like generative pretraining made large language models more adaptable 
across a wide range of tasks, even with limited management (Radford et al., 2018; Brown et al., 2020). In 
compared to other Transformer-based models, the Text-to-Text Transfer Transformer (T5) introduced a major 
change by tackling different NLP problems within a single text-to-text format (Raffel et al., 2020). This 
combined structure help to complete tasks like translation, summarization, and style transformation to be 
handled within one framework, lowering dependency on task-specific designs. However, despite these 
advances, many Transformer models are still uses and trained on formal or semi-formal data. This results that 
the model struggle with informal language, socially influenced expressions, and rapidly evolving language 
trends (Sennrich et al., 2016; Bucur et al., 2021). 
Research on informal language were there long before modern neural models came in to work and has 
continued to shape how such language is understood today. It consistently highlighted how structured and 
socially influenced informal language is. Eisenstein (2013) argued that non-standard language in online 
communication should not be dismissed as noise. Instead, it is a systematic type of language variation shaped by 
cultural and environmental factors. Later studies further supported, showing that models ignoring social 
language variation often struggle to generalize across different areas (Yang & Eisenstein, 2017). Sociolinguistic 
research also points out that informal language express‘s identity, emotion, and intent, making computational 
interpretation more complex (Hovy & Yang, 2021). Gen-Z slang is a prime example of this variation. It changes 
quickly, depends heavily on context, and meanings often changes. Regular normalization methods, like fixed 
slang dictionaries and rule-based systems, often performs poorly in these situations because they cannot adapt 
quickly or fully achieve conversational purpose (Felbo et al., 2017). Earlier methods tried to make the slang 
expressions too simple, which led to a loss of emotional part and communicative purpose. More recent work has 
293 
International Journal of Advance and Innovative Research   
Volume 13, Issue 1 (XIII): January – March 2026 
ISSN 2394 - 7780 
looked into Transformer-based solutions for informal language and slang normalization. Multilingual 
normalization models have shown better results than rule-based methods when tested on noisy and unstructured 
social media text (Bucur et al., 2021). However, in many studies, slang processing is still viewed as a one-way 
task, limiting its practical use in real conversations. Reverse generation and meaningful adaptation have not 
been fully explored. Also, language models still struggle to achieve contextual slang use, especially in limited 
resource or fast changing language environments (Camacho-Collados et al., 2022; Ishita & Mamidi, 2025). 
Some recent research pointed out a problem in neural language models and it is their transparency. Transformer 
systems mainly use black boxes, giving limited information of how linguistic transformations actually occur 
(Clark et al., 2019). When models lack transparency, it leads to no understandability and raises concerns about 
user trust, responsibility, and legal and moral use, especially when the contexts are socially sensitive (Sap et al., 
2020). Even though large language models showcase improved proficiency and contextual understanding, they 
still struggle to achieve Gen-Z intent and emotional meaning during bidirectional transformations (Camacho
Collados et al., 2022; Priccilia & Erin). This rapid evolution of digital communication makes this issue more 
complicated, as new slang terms and meanings comes into trend continuously often faster than existing datasets 
and models can adapt (Eisenstein, 2013; Hovy & Yang, 2021). In addition, scalable and efficient approaches for 
processing informal language are limited, it limits their practical deployment in conversational platforms (Yang 
& Eisenstein, 2017). A fine-tuned Transformer framework supported by improved lexical enhancement offers a 
better direction, as it allows models to understand new slangs while maintaining meaningful consistency (Bucur 
et al., 2021; Camacho-Collados et al., 2022). 
Collectively, existing research shows great progress in Transformer-based NLP, still there are several 
challenges remaining, especially in handling Gen-Z slang. Many current methods process slang as a one-way 
normalization task, provide limited understandability, and struggle to keep up with the constant evolution of 
informal digital language. These gaps point toward the need for a T5-based framework that supports 
bidirectional translation between standard English and Gen-Z slang while maintaining both meaning and Gen-Z 
intent (Raffel et al., 2020). 
3. ANALYTICAL FRAMEWORK AND RESEARCH GAP ANALYSIS 
Figure 1: Analytical Framework for Bidirectional Gen-Z Slang Translation using Transformer-based T5 Model 
Gen-Z slang translation is not like the ordinary text normalization and machine translation because it depends 
heavily on contextual understanding of words and sentence, emotional expression, and social intent instead of 
simple word to word replacement (Eisenstein, 2013; Hovy & Yang, 2021). As shown in Figure 1, these previous 
methods such as static slang dictionaries depends on fixed mappings and due to this they fail to adapt and work 
with rapidly growing slang and context-based meaning changes (Eisenstein, 2013). Rule-based normalization 
methods try to convert informal language into standard English using a set of manually defined rules but 
however, they start to struggle when the face mixed or incomplete slang usage and mainly fail to achieve, 
meaningful intent (Bucur et al., 2021). Black-box neural translation systems improve this flow using data-driven 
learning but provide limited understandability and weak control over stylistic transformation, this makes 
different slang adaptation difficult to manage and explain (Clark et al., 2019; Sap et al., 2020). These past 
limitations showed a clear research gap in understanding context, meaning understandability, and bidirectional 
slang translation systems. In addition, the availability of evolving informal datasets, such as the Gen-Z slang 
294 
International Journal of Advance and Innovative Research   
Volume 13, Issue 1 (XIII): January – March 2026 
ISSN 2394 - 7780 
sentence pairs dataset is also not easy, further highlights the need for adaptive modelling approaches (Kaggle, 
2023). To address these limitations, this proposed T5 Transformer-based framework, built on the Text-to-Text 
Transfer Transformer architecture, introduces model that can understand meaning ahead of surface-level lexical 
forms and understand the slangs (Vaswani et al., 2017; Raffel et al., 2020). This framework can translate in both 
directions, standard English and Gen-Z slang within a combined text-to-text structure, improving adaptability 
all over the linguistic contexts (Johnson et al., 2017; Man et al., 2024; Yano et al., 2024). This model includes a 
controlled lexical improvement to achieve Gen-Z intent while improving generalization in the evolving slang 
patterns (Bucur et al., 2021; Fu et al., 2018). Expressive language modeling helps to preserve emotional tone 
and conversational meaning during the translation process (Felbo et al., 2017; Sap et al., 2020). To increase 
transparency, the proposed framework combines a lexical-level explainability layer that maps word-level 
transformations and provides understandable information into stylistic adaptation processes (Clark et al., 2019; 
Fantozzi & Naldi, 2024). This part directly tackles the understandability limitations of previous neural systems 
and keeps user trust in slang translation outputs. By combining contextual semantic modelling, bidirectional text 
transformation, lexical augmentation, and explainability within Transformer architecture, the proposed model 
aligns with the analytical structure shown in Figure 1. It directly addresses the key research gap by enabling 
understandability in context, and stylistically controlled Gen-Z slang translation while preserving semantic 
meaning and emotional intent. 
4. DATA CORPUS 
The corpus used in this study is available publicly on Kaggle (Kaggle, 2023) as Gen-Z Slang Sentence Pairs. 
The original dataset has approximately 1,000 sentence pairs stored. Each pair has a standard English sentence 
and its Gen-Z slang variant. This data shows the Gen-Z slang language patterns seen in social media, direct 
messaging, and communication between young people (Eisenstein, 2013; Hovy & Yang, 2021). This results, 
that the dataset showcases current slang use, short forms, and different style of Gen-Z communication. The 
dataset represents a low and limited resource parallel corpus because, larger and consistent slang translation 
datasets are currently limited (Kaggle, 2023). This low and limited resource situation matches real-life language 
use, where slang changes quickly and labeled examples are very hard to find and they are very rare (Yang & 
Eisenstein, 2017). The sentence pairs maintain meaning-based equality but show stylistic differences, this 
ensures that the meaning stays the same across standard and slang expressions. This is important for tasks like 
slang translation and style transformation (Fu et al., 2018; Shen et al., 2017). 
In addition to the original sentence pairs, the final corpus includes extended and organized variations comes 
from the base dataset (Kaggle, 2023). These extra data and samples provide more Gen-Z slang versions for the 
current standard English sentences while keeping the original meaning intact. These variations show real 
language use, where different slang expressions can be expressed the same idea depending on context and the 
speaker‘s choice (Sap et al., 2020). Expanding the dataset in controlled way helped in increasing the number of 
aligned sentence pairs increased from about 1,000 to nearly 3,000. This helped achieve more slang variations 
while avoiding unrealistic or inconsistent examples.  The dataset is a structured parallel dataset stored in a 
comma-separated format, with each record containing two main text attributes: a standard English sentence 
labeled as ―english‖ and a Gen-Z slang sentence labeled as ―genz.‖ Each pair creates a semantically aligned 
parallel pair that supports supervised learning for tasks that have stylistic transformation. To support 
Transformer-based text-to-text modeling frameworks (Raffel et al., 2020), task-specific input and output 
formats were derived while keeping the original sentence content. For instance, an input might be structured as ―translate english to genz: this movie is very good,‖ while the corresponding output would be ―this movie is 
fire.‖ This format helps the model understand translation behavior in a text-to-text learning system. Examples of 
transformations between standard English and Gen-Z slang from the dataset are shown in Table 1. 
Table 1: Sample Sentence Pairs from the Corpus 
Standard English Sentence 
Gen-Z Slang Sentence 
this movie is fire 
This movie is very good 
I am really tired today 
My friend is coming later 
I am very excited 
That was really funny 
i‘m dead today 
my bro is pulling up later 
i‘m hella hyped 
that was funny fr 
This expanded dataset keeps semantic alignment safe while adding more stylistic alternatives, this allows 
multiple slang expression to connect to a single standard English sentence. This feature allows real-world use in 
online communication, where word choices are different while meaning remains the same (Clark et al., 2019; 
Zhang et al., 2020). 
295 
International Journal of Advance and Innovative Research   
Volume 13, Issue 1 (XIII): January – March 2026 
ISSN 2394 - 7780 
In conclusion, the final data corpus has of around 3,000 semantically aligned sentence pairs expanded from an 
initial count of around 1,000 samples. The expanding of the dataset is done with careful stylistic improvements. 
The dataset achieves a balance of realistic language, different stylistic approach, and semantic consistency.  
This makes it useful for examining Gen-Z slang translation and informal language transformation in low
resource situations (Vaswani et al., 2017; Sennrich et al., 2016). 
5. RESEARCH METHODOLOGY 
The proposed work presents a Transformer-based neural translation framework for translating between standard 
English and Gen-Z slang in both directions. The main goal of this method is to keep the meaning intact while 
allowing for controlled stylistic changes, even in low-resource situations. The system uses the fine-tuned Text
to-Text Transfer Transformer (T5-Small) architecture, which follows a typical encoder, decoder design and 
treats translation as a single text-to-text task (Raffel et al., 2020). Figure 2 shows the collective design of the 
proposed system. 
Figure 2. Architecture of the fine-tuned T5-small Transformer-based Gen-Z slang translation model. 
As shown in Figure 2, the architecture uses a Transformer-based pipeline for bidirectional translation between 
standard English and Gen-Z slang. The process starts with an input sentence that user gives and then the 
sentence is turned into subword tokens using a SentencePiece tokenizer. This helps the model deal with 
abbreviations i.e. shortform and changing slang. The tokens go to the T5 encoder. In this stage, multi-head 
attention, feed-forward layers, and normalization help the model understand the meaning of the sentence in 
context and then these components are then passed to the T5 decoder, after this it start to generate the translated 
text step by step. During this stage, masked self-attention ensures that only previously generated tokens are 
considered, while cross-attention allows the decoder to use the contextual information from the encoder outputs. 
A task-specific textual prefix decides whether to translate from English to Gen-Z or Gen-Z to English. This 
helps a single fine-tuned model work for both directions within a combined text-to-text framework (Johnson et 
al., 2017; Raffel et al., 2020). After decoding, a post-processing layer adds slang generation and normalizes 
standard English conversion. An explainability layer works with post-processing. It works with attention 
patterns and creates word-level transformation mappings between input and output tokens. This increases 
transparency while keeping the original meaning. The final translated output is made after stylistic refinement 
and interpretability analysis. Figure 3 shows the complete workflow of the system. The structure keeps meaning 
consistent while allowing some variation in style for both translation directions. Contextual embeddings created 
by the encoder help ensure the decoding keeps the intent and conversational meaning intact. Lexical addition in 
the translation improves the representation of slang but does not change the original meaning of the sentence. 
This combined process supports clear, understandable, and contextual translation of Gen-Z slang in real-world 
communication. 
296 
International Journal of Advance and Innovative Research   
Volume 13, Issue 1 (XIII): January – March 2026 
ISSN 2394 - 7780 
Figure 3. Overall workflow of the Transformer-based bidirectional Gen-Z slang translation pipeline. 
Figure 3 represents the complete workflow of the bidirectional Gen-Z slang translation system. The process 
starts by selecting the translation direction, it can be either from English to Gen-Z or from Gen-Z to English. 
Next, the text goes through preprocessing, which includes adding task-specific textual prefixes, handling noise, 
and tokenization based on the chosen translation direction. These processed tokens are then sent to the fine
tuned T5 Transformer. Then the encoder creates contextual semantic representations and the decoder generates 
the translated texts step by step. After decoding, the system applies specific post-processing based on the 
translation direction. For converting Gen-Z to English, it goes through lexical normalization. For the English to 
Gen-Z transformation, it injects controlled slang through lexical augmentation. Then the outputs are processed 
by the explainability module, which process at the token-level transformations and attention patterns. This 
creates word-level mappings between input and output. Finally, the system gives the translated sentence along 
with interpretability information, this keeps meaning and style remain correct and consistent in both translation 
directions. 
Figure 4: Architecture of the T5-Small Transformer model 
Figure 4 shows the structure of the fine-tuned T5-Small model used in this research. This architecture follows a 
Transformer-based encoder and decoder framework. Both the encoder and decoder have six stacked layers that 
are used to improve semantic understanding and translation quality (Raffel et al., 2020; Vaswani et al., 2017). 
The main process starts when an embedding layer converts tokenized input into numerical vectors. These 
vectors represent semantic meaning and similarity in context. The embeddings layers help the model to 
understand the relationships between slang expressions and their standard English variants. This helps the 
model understand the real meaning of a sentence instead of understanding each word separately and this helps 
model to keep the meaning same during the translation. Then the embedded representations go through the 
encoder stack, where multi-head self-attention tries to understand and capture the relationships between 
contextual word. The feed-forward network then refines the representations, and residual connections add the 
original input back to the output to stop information loss across layers. Layer normalization stabilizes training 
297 
International Journal of Advance and Innovative Research   
 Volume 13, Issue 1 (XIII): January – March 2026 
 
298 
ISSN 2394 - 7780 
by maintaining balanced numerical values, supporting consistent and efficient learning throughout the model. 
The encoder outputs create contextual semantic representations that are passed to the decoder through cross
attention connections. The decoder has six stacked layers as well. It includes masked self-attention, cross
attention over encoder outputs, feed-forward networks, and normalization mechanisms. Then the masked 
attention layer makes sure the decoder generates tokens one after the other and one by one by only considering 
previously generated outputs only. After this, through autoregressive process, the decoder creates the translated 
sequence while keeping semantic meaning and stylistic consistency safe with with the input text (Raffel et al., 
2020). 
This model is trained with a parallel corpus that has standard English sentences with their Gen-Z slang variants 
as a pair obtained from a public dataset (Kaggle, 2023). It ensures that the meaning stays the same across all the 
different styles. The input sentence given by the user is represented as a sequence of tokens   
            , where xᵢ is the iᵗʰ token in the sentence and n is the sequence length. SentencePiece-based 
subword tokenization converts raw text into subword units. This lets the model handle short forms, different 
spelling variations, and changing slang expressions (Sennrich et al., 2016). Given an input sequence X, the 
model produces an output sequence               , where yₜ is the token generated at time t and m is the 
output sequence length. The conditional probability of generating the target sequence is modeled as: 
    |     ∏ 
                  Equation (1) 
Equation (1) represents sequence generation as the product of conditional probabilities at each and every 
decoding step and this forms the probabilistic foundation of Transformer-based text generation (Raffel et al., 
2020; Vaswani et al., 2017). 
The encoder is responsible for semantic representation of the input sentence before the stylistic transformation. 
It contains a of stacked Transformer layers with multi-head self-attention and feed-forward networks that 
transform tokenized inputs into contextual semantic representations capturing long-range dependencies beyond 
surface lexical patterns. The self-attention mechanism is defined as: 
                        (   
√  
)      Equation (2) 
Equation (2) computes contextual relationships between tokens using query, key, and value matrices and 
enables the encoder to identify relevant semantic relationships within the sentence (Vaswani et al., 2017). The 
normalization process within the attention mechanism is defined using the softmax function: 
                
∑       
        Equation (3) 
Equation (3) shows the conversion attention scores into probabilities. It gives more importance to words that are 
relevant and important in context and less importance to those that are not. This helps the model understand 
language better and understand slang more easily (Vaswani et al., 2017). The decoder then generates the 
translated output step by step. It refers to the words that are previously generated using the masked self
attention and it also uses the information from the encoder through cross-attention. This connection between the 
input and output helps to correct translated outputs while keeping the original meaning intact and with less 
Noice (Johnson et al., 2017). The T5-small model contains six encoder layers and six decoder layers, and each 
layer uses multi-head attention and feed-forward networks to understand patterns in language and understand 
context all over the sentence. Positional information is added so that the model can maintain the correct order of 
words during processing (Vaswani et al., 2017). After decoding, lexical augmentation is applied to manage 
stylistic changes. The neural model first creates context-aware translations, and then a lexical module adjusts 
slang expressions when converting English to Gen-Z language and applies normalization when translating in the 
opposite direction. This separation allows the neural model to focus on meaning while the lexical component 
controls the style (Bucur et al., 2021; Fu et al., 2018). Normalization is use to remove noise, understand short 
forms, reduce confusion, and maintain the original meaning of the sentence. The lexical step ensures sentence 
accuracy without altering the core and original message. An explainability layer is then used to understand how 
the translation was produced. It studies attention patterns between the encoder and decoder to identify how input 
words influence the generated output (Clark et al., 2019; Sap et al., 2020; Fantozzi & Naldi, 2024). Attention 
information from different heads is combined to create word-level mappings and show how transformations 
occur. This makes the translation process more transparent and easier to interpret. 
This model learns meaning through contextual attention rather than relying on predefined semantic rules. It 
captures language relationships directly from data using representation learning (Vaswani et al., 2017; Peters et 
International Journal of Advance and Innovative Research   
Volume 13, Issue 1 (XIII): January – March 2026 
ISSN 2394 - 7780 
al., 2018). The model training done by standard Transformer fine-tuning practices. T5-small Transformer is 
selected for balance performance and efficiency. Both encoder and decoder contain six layers with multi-head 
attention and a hidden dimension of 512. The sequence length is set according to conversational text 
requirements. Training is done by using the AdamW as optimizer with a learning rate of 5×10⁻⁵, batch size of 
eight, and five epochs to ensure stable learning in a low-resource setting (Raffel et al., 2020). This methodology 
combines contextual Transformer modeling, bidirectional translation, lexical augmentation, normalization, and 
explainability within a single framework. This approach keeps the meaning same, allows controlled stylistic 
changes, and produces clear translations suited for real-world Gen-Z slang communication. 
6. RESULTS AND DISCUSSION 
This proposed Transformer-based Gen-Z slang translation system was evaluated using the BLEU (Bilingual 
Evaluation Understudy) score, a widely used metric for evaluating neural text generation. BLEU catches 
overlap between generated and reference text while still allowing acceptable variations in wording. This is 
especially useful for slang translation, where different expressions can express the same meaning (Hovy & 
Yang, 2021; Fu et al., 2018). This model achieved a BLEU score of 0.43, which is strong for a low and limited 
resource dataset containing informal, evolving, and context dependent language. Instead of only matching 
surface-level words, this score shows the model‘s ability to preserve meaning while performing controlled 
stylistic changes. Similar improvements in T5-based translation and multilingual modeling have been covered 
in recent studies (Zhu et al., 2025; Yano et al., 2024).  
A comparison of different model configurations is shown in Table 2, this table shows performance after fine
tuning of the model and expanding the data. The performance is improved after lexical augmentation; this 
shows that controlled stylistic adaptation plays an important role in informal language translation. Similar 
results have been seen in earlier research on lexical normalization and style-aware neural translation. (Fu et al., 
2018; Bucur et al., 2021). 
Table 2. BLEU score performance comparison 
Model Configuration 
Dataset Size 
Baseline T5 (no fine-tuning) 
Fine-tuned T5 Transformer 
Fine-tuned T5 + lexical 
augmentation (proposed) 
BLEU Score 
— 
~1K samples 
~3.5K samples 
(augmented) 
Low / 
Inconsistent 
0.38 
0.43 
In addition to quantitative evaluation, the observations showed that the system consistently preserved meaning 
while generating correct slang expressions that are appropriate for the context. This model avoids unnecessary 
changes when slang is not suitable, which help the model to maintain grammatical clarity and semantic 
consistency. It performed great in both translated directions. It generated Gen-Z slang from standard English 
and converted slang into clear, standard English using the text-to-text framework. Understandability was also 
considered in the evaluation. The system has an attention-based analysis component that provides word-level 
transformation mappings, token relationships, and slang-to-standard correspondences. These features support 
debugging, qualitative assessment, and use in educational or socially sensitive contexts (Clark et al., 2019; Sap 
et al., 2020; Fantozzi & Naldi, 2024). 
From a practical point of view, the Transformer-based architecture supports scalability and easy updates. The 
parallel low and limited resource conditions benefit the T5 model and this makes it suitable for deployment in 
real-world applications (Vaswani et al., 2017; Raffel et al., 2020; Brown et al., 2020). The lexical augmentation 
module allows new slang expressions to be added to the neural model. This separation between contextual 
learning and lexical control improves adaptability in constantly changing language environments. It aligns with 
previous research on informal language processing and social media text normalization (Bucur et al., 2021; 
Camacho-Collados et al., 2022).  
In the end, the results show that the proposed T5-based framework performs well in Gen-Z slang translation in 
low and limited resource settings. This system maintains semantic meaning, applies correct stylistic 
transformations, provides understandability, and remains scalable for real-world use. These features make it 
suitable for applications such as conversational AI, educational tools, and social media language analysis. 
7. CONCLUSION 
In this work, a Transformer-based framework was developed for translating between standard English and Gen
Z slang under low and limited resource conditions. This system used a T5-Small encoder–decoder architecture 
within a combined text-to-text setting, enabling translation in both directions while keep the sentence meaning 
safe and conversational intent. The research focused on balancing semantic consistency under a controlled 
299 
International Journal of Advance and Innovative Research   
Volume 13, Issue 1 (XIII): January – March 2026 
ISSN 2394 - 7780 
stylistic variation, which is very important for informal and socially influenced language translation. This 
evaluation shows that the proposed model achieved a BLEU score of 0.43 on the expanded Gen-Z slang dataset, 
indicating a great alignment between original and translated text even with the challenges of informal, context
dependent language. The combination of lexical augmentation supported flexible slang generation and 
normalization, while the attention-based explainability component help us to understand word-level 
transformation insights and information that improves transparency and understandability of translation outputs. 
This architecture also supports practical deployment in evolving linguistic environments. The separation 
between contextual semantic modeling and lexical control allows new slang patterns to be added without any 
problem, making the system adaptable to rapid language change. These capabilities position the framework as a 
useful approach for real-world applications such as conversational systems, social media text processing, and 
informal language normalization. The study also talks about the importance of combining contextual modeling, 
conversational control, and understandability when working with informal digital language. Transformer-based 
systems may achieve fluency, but effective slang translation needs attention to emotional tone, conversational 
intent, and explainability of linguistic transformations. By combining these components within a single 
architecture, the proposed method contributes toward more transparent and socially aware NLP systems. This 
work establishes a foundation for future exploration in explainable slang translation and context-aware informal 
language processing, particularly in low-resource scenarios where linguistic variation evolves rapidly (Raffel et 
al., 2020; Vaswani et al., 2017; Bucur et al., 2021; Fantozzi & Naldi, 2024). 
8. FUTURE WORK AND RECOMMENDATIONS 
Future work can focus on increasing the dataset in more different and context-specific slang to improve model 
dependency and ease of use with changing informal language. Focus should be on increasing the variety of the 
corpus to maintain wider stylistic patterns and growing slang expressions in different digital communication 
settings. Using larger or more efficient Transformer models may improve translation quality while keeping 
computational efficiency. The understandability aspect can be extended by using attention-based analysis 
techniques to provide deeper information into translation behavior and improve transparency.  Future efforts 
will aim to strengthen bidirectional translation capability. This will enable more accurate and context-sensitive 
conversion between standard English and Gen-Z slang. Taking user feedback, keep track of real-time slang 
updates and combining both together can help with ongoing system improvement and practical use. These 
improvements will contribute to creating more flexible, scalable, and understandable Gen-Z slang translation 
systems based on fine-tuned Transformer architectures and text-to-text learning frameworks (Raffel et al., 2020; 
Vaswani et al., 2017; Brown et al., 2020). 
Acknowledgement 
The authors sincerely thank the Department of Information Technology, Sheth L.U. Jhaveri College of Arts, and 
Sir M.V. College of Science and Commerce for their support, guidance, and resources that helped in completing 
this research work 
Ethical Statement 
This study did not involve experiments with human participants or animal subjects. All datasets used in this 
research were obtained from publicly available sources and processed in accordance with ethical research 
guidelines. 
Conflicts of Interest 
The authors declare that there are no conflicts of interest related to this research work. 
Data Availability Statement 
The dataset used in this study was sourced from publicly available Gen-Z slang datasets. The processed data 
and trained model artifacts can be provided by the authors upon reasonable request. 
REFERENCES 
Kaggle. 
(2023). 
Gen-Z 
slang 
sentence 
pairs 
dataset 
(Version 
1) 
[Dataset]. 
Kaggle. 
https://www.kaggle.com/datasets/programmerrdai/genz-slang-pairs-1k 
Bucur, A.-M., Cosma, A., & Dinu, L. P. (2021). Sequence-to-sequence lexical normalization with multilingual 
transformers. https://arxiv.org/abs/2110.02869 
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, 
G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., 
Wu, J., Winter, C., & Amodei, D. (2020). Language models are few-shot learners. Advances in Neural 
Information Processing Systems. https://arxiv.org/abs/2005.14165 
300 
International Journal of Advance and Innovative Research   
Volume 13, Issue 1 (XIII): January – March 2026 
ISSN 2394 - 7780 
Camacho-Collados, J., Rezaee, K., Riahi, T., Ushio, A., Loureiro, D., Antypas, D., Boisson, J., Espinosa-Anke, 
L., Liu, F., Martínez-Cámara, E., Medina, G., Buhrmann, T., Neves, L., & Barbieri, F. (2022). TweetNLP: 
Cutting-edge natural language processing for social media. https://arxiv.org/abs/2206.14774 
Chen, Z., Qiu, Z., & Chen, N. (2024). Bidirectional multi-stack RNNs with attention for machine translation. 
IEEE. https://ieeexplore.ieee.org/abstract/document/10430889/authors#authors 
Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What does BERT look at? An analysis of 
BERT‘s attention. https://arxiv.org/abs/1906.04341 
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional 
transformers for language understanding. https://aclanthology.org/N19-1423 
Diwan, A., Choi, E., & Harwath, D. (2023). When to use efficient self-attention? Profiling text, speech and 
image transformer variants. arXiv. https://arxiv.org/pdf/2306.08667 
Eisenstein, J. (2013). What to do about bad language on the internet. https://aclanthology.org/N13-1037.pdf 
Fantozzi, P., & Naldi, M. (2024). The explainability of transformers: Current status and directions. Computers. 
https://www.mdpi.com/2073-431X/13/4/92 
any-domain 
Felbo, B., Mislove, A., Søgaard, A., Rahwan, I., & Lehmann, S. (2017). Using millions of emoji occurrences to 
learn 
representations 
for 
detecting 
sentiment, 
emotion 
and 
sarcasm. 
https://arxiv.org/abs/1708.00524 
Fu, Z., Tan, X., Peng, N., Zhao, D., & Yan, R. (2018). Style transfer in text: Exploration and evaluation. 
https://arxiv.org/abs/1711.06861 
Gehring, J., Auli, M., Grangier, D., Yarats, D., & Dauphin, Y. N. (2017). Convolutional sequence to sequence 
learning. https://arxiv.org/abs/1705.03122 
Hovy, D., & Yang, D. (2021). The importance of modeling social factors of language: Theory and practice. 
https://aclanthology.org/2021.naacl-main.49 
Ibrahim, A. A., & Mustafa, B. S. (2023). Intelligent system to transform slang words into formal words. NTU 
Journal for Engineering and Technology. https://journals.ntu.edu.iq/index.php/NTU-JET/article/view/689 
Ishita, & Mamidi, R. (2025). The evolution of Gen Alpha slang: Linguistic patterns and AI translation 
challenges. https://aclanthology.org/anthology-files/pdf/acl/2025.acl-srw.43.pdf 
Johnson, M., Schuster, M., Le, Q. V., Krikun, M., Wu, Y., Chen, Z., Thorat, N., Vi gas, F., Wattenberg, M., 
Corrado, G., Hughes, M., & Dean, J. (2017). Google‘s multilingual neural machine translation system: Enabling 
zero-shot translation. https://arxiv.org/abs/1611.04558 
Kadem, M., & Zheng, R. (2026). Interpreting transformers through attention head intervention. arXiv. 
https://arxiv.org/abs/2601.04398 
Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. 
(2020). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and 
comprehension. https://aclanthology.org/2020.acl-main.703/ 
Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. 
(2019). RoBERTa: A robustly optimized BERT pretraining approach. https://arxiv.org/abs/1907.11692 
Man, C. Z., Win, S. S. M., & Khine, K. L. L. (2024). Fine-tuning the mT5 model on bidirectional Myanmar and 
Tedim Chin machine translation system. International Journal of Computer Science. 
https://www.iaeng.org/IJCS/issues_v52/issue_12/IJCS_52_12_03.pdf 
Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep 
contextualized word representations. https://arxiv.org/abs/1802.05365 
Pew Research Center. (2019). Where Millennials end and Generation Z begins. 
https://www.pewresearch.org/short-reads/2019/01/17/where-millennials-end-and-generation-z-begins/ 
Priccilia, S., & Erin. Sentiment analysis of slang language trends in Generation Alpha on social media using 
BERT. https://doi.org/10.21512/emacsjournal.v7i2.13368 
301 
International Journal of Advance and Innovative Research   
Volume 13, Issue 1 (XIII): January – March 2026 
ISSN 2394 - 7780 
Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by 
generative 
pre-training. 
https://cdn.openai.com/research-covers/language
unsupervised/language_understanding_paper.pdf 
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). 
Exploring the limits of transfer learning with a unified text-to-text transformer. https://arxiv.org/abs/1910.10683 
Sap, M., Gabriel, S., Qin, L., Jurafsky, D., Smith, N. A., & Choi, Y. (2020). Social bias frames: Reasoning 
about social and power implications of language. https://arxiv.org/abs/1911.03891 
Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. 
https://arxiv.org/abs/1508.07909 
Shen, J., Qu, Y., Zhang, W., & Yu, Y. (2017). Style transfer from non-parallel text by cross-alignment. 
https://arxiv.org/abs/1705.09655 
Findings 
Song, L., Cui, Y., Luo, A., Lecue, F., & Li, I. (2024). Better explain transformers by illuminating important 
information. 
of 
the 
Association 
for 
Computational 
Linguistics 
(ACL). 
https://aclanthology.org/2024.findings-eacl.138/ 
Tang, L. (2024). Optimization of English machine translation based on bidirectional encoder representations 
from 
transformers-integrated 
neural 
https://ieeexplore.ieee.org/abstract/document/11070579/authors#authors 
architectures. 
IEEE. 
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. 
(2017). Attention is all you need. https://arxiv.org/abs/1706.03762 
Yang, W., & Eisenstein, J. (2017). Overcoming language variation in sentiment analysis with social attention. 
https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00062/43395 
Yano, C., Fukuchi, A., Fukasawa, S., Tachibana, H., & Watanabe, Y. (2024). Multilingual Sentence-T5: 
Scalable sentence encoders for multilingual applications. Proceedings of LREC-COLING 2024. 
https://aclanthology.org/2024.lrec-main.1034/ 
Zaki, M. Z. (2023). Revolutionising translation technology: A comparative study of variant transformer 
models—BERT, 
GPT 
and 
T5. 
https://www.academia.edu/download/116199770/Published_Revolutionising_Trans_Tech.pdf 
Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., & Weston, J. (2020). DialoGPT: Large-scale generative 
pre-training for conversational response generation. https://aclanthology.org/2020.acl-demos.30/ 
Zhu, J., Sun, H., & Kong, B. (2025). Improving multilingual English translation performance through T5 and 
MAML 
integration. 
Results 
https://www.sciencedirect.com/science/article/pii/S2772941925002133 
in 
Engineering. 