import { Shield } from 'lucide-react'
import { LearningRoadmap } from '../LearningRoadmap'
import type { LearningTopic } from '@/lib/types'

const INITIAL_SECURITY_TOPICS: LearningTopic[] = [
  {
    id: 'sec-1',
    title: 'Cybersecurity Fundamentals',
    description: 'Introduction to the CIA Triad (Confidentiality, Integrity, Availability), threat actors, and basic risk management.',
    category: 'Fundamentals',
    difficulty: 'Beginner',
    estimatedTime: '2 hours',
    completed: true,
    content: ''
  },
  {
    id: 'sec-2',
    title: 'Authentication & Access Control',
    description: 'Learn about MFA, OAuth, SSO, RBAC (Role-Based Access Control), and password hashing algorithms.',
    category: 'Identity',
    difficulty: 'Beginner',
    estimatedTime: '2.5 hours',
    completed: false,
    content: ''
  },
  {
    id: 'sec-3',
    title: 'Network Security',
    description: 'Understand firewalls, IDS/IPS, VPNs, subnetting security, and common network vulnerabilities.',
    category: 'Infrastructure',
    difficulty: 'Intermediate',
    estimatedTime: '3 hours',
    completed: false,
    content: ''
  },
  {
    id: 'sec-4',
    title: 'Common Cyber Attacks (OWASP)',
    description: 'Explore SQL Injection, Cross-Site Scripting (XSS), CSRF, and how to mitigate the OWASP Top 10 web vulnerabilities.',
    category: 'AppSec',
    difficulty: 'Intermediate',
    estimatedTime: '4 hours',
    completed: false,
    content: ''
  },
  {
    id: 'sec-5',
    title: 'Cryptography & Encryption',
    description: 'Deep dive into symmetric vs asymmetric encryption, PKI, digital signatures, and TLS/SSL handshakes.',
    category: 'Advanced',
    difficulty: 'Advanced',
    estimatedTime: '4.5 hours',
    completed: false,
    content: ''
  }
]

export function SecurityLearningView() {
  return (
    <LearningRoadmap 
      title="Security Learning" 
      icon={<Shield size={24} />} 
      description="Master cybersecurity concepts, defensive strategies, and ethical hacking fundamentals."
      initialTopics={INITIAL_SECURITY_TOPICS}
      colorTone="critical"
    />
  )
}
