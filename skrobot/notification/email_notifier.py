import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from skrobot.notification import BaseNotifier

class EmailNotifier(BaseNotifier):
  """
  The :class:`.EmailNotifier` can be used to send email notifications of successfull/failure task to some recipients.

  """
  def __init__ (self,  sender_account, sender_username, sender_pass, smtp_server, smtp_port, recipients):
    """
	This is the constructor method and can be used to create a new object instance of :class:`.EmailNotifier` class.	
	
	:param sender_account: The email account of the sender. For example "someone@gmail.com"
    :type sender_account: str
	
	:param sender_username: The username of the sender email account. For example "someone" is the username to email account "someone@gmail.com"  
	:type sender_username: str
	
	:param sender_pass: The password of the sender email account. 
	:type sender_pass: str 
	
	:param smtp_server: The smtp server of the sender email account. For example for gmail is "smtp.gmail.com".
	:type smtp_server: str
	
	:param smtp_port: The port of the smtp server. For example for gmail accounts is 587
    :type smtp_port: int
	
	:param recipients: The list with the recipients of email notifications.
	:type recipients: list 
	"""
    self.sender_account = sender_account 
    self.sender_username = sender_username
    self.sender_pass = sender_pass
    self.smtp_server = smtp_server
    self.smtp_port = smtp_port
    self.recipients = recipients
	
  def notify(self, message):
    """
    The method sends the email notification to the recipients.

    :param message: The notification's message.
    :type message: str
    """
    email_subject = "Skrobot Notification"
    server = smtplib.SMTP(self.smtp_server,self.smtp_port)
    server.starttls()
    server.login(self.sender_username, self.sender_pass)#For loop, sending emails to all email recipients
    for recipient in self.recipients:
      email = MIMEMultipart('alternative')
      email['From'] = self.sender_account
      email['To'] = recipient
      email['Subject'] = email_subject
      email.attach(MIMEText(message, 'html'))
      text = email.as_string()
      server.sendmail(self.sender_account,recipient,text)
    server.quit()
  

