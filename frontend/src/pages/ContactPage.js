import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  CircularProgress
} from '@mui/material';
import {
  Email as EmailIcon,
  Phone as PhoneIcon,
  LocationOn as LocationIcon,
  AccessTime as TimeIcon,
  Send as SendIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import { toast } from 'react-toastify';
import axios from 'axios';

const ContactPage = () => {
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const validationSchema = Yup.object({
    name: Yup.string()
      .min(2, 'Name must be at least 2 characters')
      .required('Name is required'),
    email: Yup.string()
      .email('Invalid email format')
      .required('Email is required'),
    subject: Yup.string()
      .min(5, 'Subject must be at least 5 characters')
      .required('Subject is required'),
    message: Yup.string()
      .min(10, 'Message must be at least 10 characters')
      .required('Message is required')
  });

  const formik = useFormik({
    initialValues: {
      name: '',
      email: '',
      subject: '',
      message: ''
    },
    validationSchema,
    onSubmit: async (values, { resetForm }) => {
      setLoading(true);
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 2000));

        setSubmitted(true);
        resetForm();
        toast.success('Message sent successfully! We\'ll get back to you soon.');
      } catch (error) {
        toast.error('Failed to send message. Please try again.');
      } finally {
        setLoading(false);
      }
    }
  });

  const contactInfo = [
    {
      icon: <EmailIcon color="primary" />,
      title: 'Email Us',
      details: 'support@strokepredictor.com',
      description: 'We typically respond within 24 hours'
    },
    {
      icon: <PhoneIcon color="primary" />,
      title: 'Call Us',
      details: '+1 (555) 123-4567',
      description: 'Monday - Friday, 9:00 AM - 6:00 PM EST'
    },
    {
      icon: <LocationIcon color="primary" />,
      title: 'Visit Us',
      details: '123 Health Tech Ave, Medical District',
      description: 'San Francisco, CA 94105'
    },
    {
      icon: <TimeIcon color="primary" />,
      title: 'Support Hours',
      details: '24/7 Online Support',
      description: 'Emergency support available anytime'
    }
  ];

  const faqs = [
    {
      question: 'How accurate is the stroke risk prediction?',
      answer: 'Our AI model achieves 99.5% accuracy based on clinical validation studies. However, this should complement, not replace, professional medical advice.'
    },
    {
      question: 'Is my health data secure?',
      answer: 'Yes, we use bank-level encryption and are HIPAA compliant. Your data is never shared with third parties without your explicit consent.'
    },
    {
      question: 'How often should I take the assessment?',
      answer: 'We recommend taking the assessment every 6-12 months, or whenever there are significant changes in your health status or lifestyle.'
    },
    {
      question: 'Can I use this tool if I\'m under 18?',
      answer: 'Our tool is designed for adults 18 and older. Minors should consult with their pediatrician for appropriate health assessments.'
    }
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography
          variant="h3"
          component="h1"
          fontWeight="bold"
          sx={{ mb: 2, fontSize: { xs: '2rem', md: '3rem' } }}
        >
          📞 Contact Us
        </Typography>
        <Typography
          variant="h6"
          color="textSecondary"
          sx={{ maxWidth: 600, mx: 'auto' }}
        >
          Have questions about our stroke risk prediction system?
          We're here to help and would love to hear from you.
        </Typography>
      </Box>

      <Grid container spacing={6}>
        {/* Contact Form */}
        <Grid item xs={12} md={8}>
          <Card elevation={2}>
            <CardContent sx={{ p: 4 }}>
              <Typography variant="h5" fontWeight="bold" sx={{ mb: 3 }}>
                Send us a Message
              </Typography>

              {submitted ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <CheckCircleIcon sx={{ fontSize: 80, color: 'success.main', mb: 2 }} />
                  <Typography variant="h6" color="success.main" sx={{ mb: 2 }}>
                    Message Sent Successfully!
                  </Typography>
                  <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
                    Thank you for contacting us. We'll get back to you within 24 hours.
                  </Typography>
                  <Button
                    variant="outlined"
                    onClick={() => setSubmitted(false)}
                  >
                    Send Another Message
                  </Button>
                </Box>
              ) : (
                <Box component="form" onSubmit={formik.handleSubmit}>
                  <Grid container spacing={3}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        id="name"
                        name="name"
                        label="Your Name"
                        value={formik.values.name}
                        onChange={formik.handleChange}
                        onBlur={formik.handleBlur}
                        error={formik.touched.name && Boolean(formik.errors.name)}
                        helperText={formik.touched.name && formik.errors.name}
                        disabled={loading}
                      />
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        id="email"
                        name="email"
                        label="Email Address"
                        type="email"
                        value={formik.values.email}
                        onChange={formik.handleChange}
                        onBlur={formik.handleBlur}
                        error={formik.touched.email && Boolean(formik.errors.email)}
                        helperText={formik.touched.email && formik.errors.email}
                        disabled={loading}
                      />
                    </Grid>

                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        id="subject"
                        name="subject"
                        label="Subject"
                        value={formik.values.subject}
                        onChange={formik.handleChange}
                        onBlur={formik.handleBlur}
                        error={formik.touched.subject && Boolean(formik.errors.subject)}
                        helperText={formik.touched.subject && formik.errors.subject}
                        disabled={loading}
                      />
                    </Grid>

                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        id="message"
                        name="message"
                        label="Your Message"
                        multiline
                        rows={6}
                        value={formik.values.message}
                        onChange={formik.handleChange}
                        onBlur={formik.handleBlur}
                        error={formik.touched.message && Boolean(formik.errors.message)}
                        helperText={formik.touched.message && formik.errors.message}
                        disabled={loading}
                      />
                    </Grid>

                    <Grid item xs={12}>
                      <Button
                        type="submit"
                        variant="contained"
                        size="large"
                        fullWidth
                        startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
                        disabled={loading || !formik.isValid}
                        sx={{ py: 1.5 }}
                      >
                        {loading ? 'Sending...' : 'Send Message'}
                      </Button>
                    </Grid>
                  </Grid>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Contact Information */}
        <Grid item xs={12} md={4}>
          <Grid container spacing={3}>
            {contactInfo.map((info, index) => (
              <Grid item xs={12} key={index}>
                <Card>
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
                      {info.icon}
                      <Box sx={{ ml: 2 }}>
                        <Typography variant="h6" fontWeight="bold">
                          {info.title}
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 1 }}>
                          {info.details}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          {info.description}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>
      </Grid>

      {/* FAQ Section */}
      <Box sx={{ mt: 8 }}>
        <Typography variant="h4" fontWeight="bold" sx={{ mb: 4, textAlign: 'center' }}>
          Frequently Asked Questions
        </Typography>
        <Grid container spacing={3}>
          {faqs.map((faq, index) => (
            <Grid item xs={12} md={6} key={index}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" fontWeight="bold" sx={{ mb: 2 }}>
                    {faq.question}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    {faq.answer}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Support Notice */}
      <Paper
        sx={{
          mt: 6,
          p: 4,
          backgroundColor: 'info.light',
          border: '1px solid',
          borderColor: 'info.main'
        }}
      >
        <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: 'info.dark' }}>
          🚨 Emergency Medical Situations
        </Typography>
        <Typography variant="body1" color="info.dark">
          If you are experiencing symptoms of a stroke or any medical emergency,
          do not use this contact form. Call emergency services immediately (911 in the US)
          or go to your nearest emergency room.
        </Typography>
        <Typography variant="body2" sx={{ mt: 2, fontStyle: 'italic' }}>
          Remember: F.A.S.T. - Face drooping, Arm weakness, Speech difficulties, Time to call emergency services.
        </Typography>
      </Paper>

      {/* Additional Resources */}
      <Box sx={{ mt: 6, textAlign: 'center' }}>
        <Typography variant="h5" fontWeight="bold" sx={{ mb: 3 }}>
          Other Ways to Connect
        </Typography>
        <Typography variant="body1" color="textSecondary" sx={{ mb: 3 }}>
          Join our community and stay updated with the latest in stroke prevention and health technology.
        </Typography>
        <Grid container spacing={2} justifyContent="center">
          <Grid item>
            <Button variant="outlined" href="/blog" target="_blank">
              Read Our Blog
            </Button>
          </Grid>
          <Grid item>
            <Button variant="outlined" href="/newsletter" target="_blank">
              Subscribe to Newsletter
            </Button>
          </Grid>
          <Grid item>
            <Button variant="outlined" href="/community" target="_blank">
              Join Community Forum
            </Button>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default ContactPage;
