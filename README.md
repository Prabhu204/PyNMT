# PyNMT
<h2>Neural machine translator form English to German</h2>
NMT model is composed of one encoder for source language, one decoder for target language. However, the data required for
building such model requires language pair phrases. The following steps will explain NMT models.  
<h3>1. Preparing the Text Data</h3>
Before tokenization of data, it will be cleaned with following below mentioned steps.

    Remove all non-printable characters.
    Remove all punctuation characters.
    Normalize all Unicode characters to ASCII (e.g. Latin characters).
    Normalize the case to lowercase.
    Remove any remaining tokens that are not alphabetic.
   




https://drive.google.com/drive/folders/1bSZtJAeMIVdhtxBQgSn7BgRzrF-05ZnX?usp=sharing