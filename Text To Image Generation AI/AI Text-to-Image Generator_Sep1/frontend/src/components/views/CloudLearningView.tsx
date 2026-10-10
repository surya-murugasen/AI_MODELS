import { Cloud } from 'lucide-react'
import { LearningRoadmap } from '../LearningRoadmap'
import type { LearningTopic } from '@/lib/types'

const INITIAL_CLOUD_TOPICS: LearningTopic[] = [
  {
    id: 'cloud-1',
    title: 'Cloud Computing Fundamentals',
    description: 'Understand the basics of cloud computing, deployment models (Public, Private, Hybrid), and service models (IaaS, PaaS, SaaS).',
    category: 'Fundamentals',
    difficulty: 'Beginner',
    estimatedTime: '2 hours',
    completed: true,
    content: ''
  },
  {
    id: 'cloud-2',
    title: 'AWS Global Infrastructure',
    description: 'Learn about AWS Regions, Availability Zones, Edge Locations, and how they provide high availability and fault tolerance.',
    category: 'Architecture',
    difficulty: 'Beginner',
    estimatedTime: '1.5 hours',
    completed: false,
    content: ''
  },
  {
    id: 'cloud-3',
    title: 'Elastic Compute Cloud (EC2)',
    description: 'Deep dive into Amazon EC2 instance types, AMIs, Security Groups, and connecting to instances via SSH.',
    category: 'Compute',
    difficulty: 'Intermediate',
    estimatedTime: '3 hours',
    completed: false,
    content: ''
  },
  {
    id: 'cloud-4',
    title: 'S3 & Cloud Storage',
    description: 'Explore Amazon Simple Storage Service (S3), storage classes, versioning, lifecycle policies, and bucket security.',
    category: 'Storage',
    difficulty: 'Intermediate',
    estimatedTime: '2.5 hours',
    completed: false,
    content: ''
  },
  {
    id: 'cloud-5',
    title: 'Serverless Architecture (Lambda)',
    description: 'Introduction to serverless computing, creating AWS Lambda functions, triggers, and API Gateway integration.',
    category: 'Advanced Compute',
    difficulty: 'Advanced',
    estimatedTime: '4 hours',
    completed: false,
    content: ''
  }
]

export function CloudLearningView() {
  return (
    <LearningRoadmap 
      title="Cloud Learning" 
      icon={<Cloud size={24} />} 
      description="Master cloud computing, AWS infrastructure, and modern deployment strategies."
      initialTopics={INITIAL_CLOUD_TOPICS}
      colorTone="blue"
    />
  )
}
