import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatApplicationComponent } from './chat-application.component';

describe('ChatApplicationComponent', () => {
  let component: ChatApplicationComponent;
  let fixture: ComponentFixture<ChatApplicationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatApplicationComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ChatApplicationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
