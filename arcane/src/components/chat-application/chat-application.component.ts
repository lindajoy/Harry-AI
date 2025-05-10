import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderGenerationComponent } from '../header-generation/header-generation.component';
import { FormControl, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { ChatService } from '../../services/chat.service';

@Component({
  selector: 'app-chat-application',
  standalone: true,
  imports: [CommonModule, FormsModule, ReactiveFormsModule],
  templateUrl: './chat-application.component.html',
  styleUrl: './chat-application.component.css'
})
export class ChatApplicationComponent {
  chatService = inject(ChatService);
  answer: string = '';

  formGroup = new FormGroup({
    prompt: new FormControl('', Validators.required),
    tone: new FormControl('', Validators.required),
    includeSources: new FormControl(false),
  });

  tones = [
    { value: "professional", label: "Professional" },
    { value: "genz", label: "Gen Z" },
    { value: "casual", label: "Casual" },
    { value: "academic", label: "Academic" },
  ];

  atitudes = [
    { value: "wizard", label: "Wizard" },
    { value: "sacarstic", label: "sacarstic" },
    { value: "quirky", label: "quirky" },
    { value: "Too cool to care", label: "Too cool to care" },
    { value: "funny", label: "Funny" },
  ]

  generateResponse() {
    const formValue = this.formGroup.value ?? {};
    this.chatService.generateResponse(formValue).subscribe((response: any) => {
       this.answer = response.response;
   })
  }

  autoResize(event: Event): void {
    const textarea = event.target as HTMLTextAreaElement;
    textarea.style.height = 'auto'; // reset first
    textarea.style.height = textarea.scrollHeight + 'px';
  }
  
}
